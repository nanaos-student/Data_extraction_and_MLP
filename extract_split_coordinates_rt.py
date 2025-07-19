import os
import sys
import traci
import sumolib
import numpy as np
import pandas as pd
from collections import defaultdict


def process_simulation(net_file, sumo_cfg,
                       global_A_csv_prefix='zone_A_',
                       global_B_csv_prefix='zone_B_',
                       total_simulation_time=600,  # 10 minutes
                       output_interval=60):  # output every 60 seconds
    # Nested function to check if required files exist
    def check_files_exist(net_file, sumo_cfg):
        if not os.path.exists(net_file):
            raise FileNotFoundError(f"Network file not found: {net_file}")
        if not os.path.exists(sumo_cfg):
            raise FileNotFoundError(f"SUMO config file not found: {sumo_cfg}")

    # Nested function to split DataFrame into two based on the zone column
    def splitter(dataframe):
        mask_A = (dataframe['zone'] == 'A') | (dataframe['zone'] == 'border')
        filtered_A = dataframe[mask_A]
        mask_B = (dataframe['zone'] == 'B') | (dataframe['zone'] == 'border')
        filtered_B = dataframe[mask_B]
        global_A = filtered_A.copy()
        global_B = filtered_B.copy()
        return global_A, global_B

    # Verify that the required files exist
    check_files_exist(net_file, sumo_cfg)

    # Read the network file and compute spatial boundaries
    net = sumolib.net.readNet(net_file)
    nodes = net.getNodes()
    x_coords = [node.getCoord()[0] for node in nodes]
    y_coords = [node.getCoord()[1] for node in nodes]
    max_x = max(x_coords)
    min_x = min(x_coords)
    max_y = max(y_coords)
    min_y = min(y_coords)
    split_x = (min_x + max_x) / 2

    # Filter vehicular edges by allowed highway types
    allowed_types = {
        'highway.motorway', 'highway.motorway_link',
        'highway.trunk', 'highway.trunk_link',
        'highway.primary', 'highway.primary_link',
        'highway.secondary', 'highway.secondary_link',
        'highway.tertiary', 'highway.tertiary_link',
        'highway.unclassified', 'highway.residential',
        'highway.bus_guideway'
    }
    edge_ids = [
        edge.getID()
        for edge in net.getEdges()
        if (not edge.getID().startswith(":") and edge.getType() in allowed_types)
    ]

    # Create a zone dictionary using the split_x value
    zone_dict = {}
    for edge in net.getEdges():
        if edge.getID() not in edge_ids:
            continue
        from_x = edge.getFromNode().getCoord()[0]
        to_x = edge.getToNode().getCoord()[0]
        if from_x < split_x and to_x < split_x:
            zone_dict[edge.getID()] = 'A'
        elif from_x >= split_x and to_x >= split_x:
            zone_dict[edge.getID()] = 'B'
        else:
            zone_dict[edge.getID()] = 'border'

    # Setup SUMO command for simulation
    sumo_cmd = [
        "sumo",
        "-c", sumo_cfg,
        "--no-warnings",
        "--begin", "0",
        "--end", str(total_simulation_time),
        "--step-length", "0.5",
        "--device.emissions.probability", "1.0"
    ]

    # Data containers for simulation metrics
    emission_data = defaultdict(lambda: {
        'co2': [], 'co': [], 'hc': [], 'nox': [],
        'pmx': [], 'fuel': [], 'noise': []
    })
    travel_times = defaultdict(list)
    vehicle_counts = defaultdict(list)
    speeds = defaultdict(list)
    zone_data_history = []

    try:
        traci.start(sumo_cmd)
        steps_per_output = output_interval * 2  # 2 steps per second
        print("\nStarting simulation...")

        # Retrieve traffic lights at the start
        traffic_lights = traci.trafficlight.getIDList()

        for step in range(0, total_simulation_time + 1, 2):  # 2 steps per second
            if step % 120 == 0:
                print(f"Progress: {step // 120} minutes of {total_simulation_time // 60} minutes")

            traci.simulationStep()

            # Record metrics every 15 steps
            if step % 15 == 0:
                for edge_id in edge_ids:
                    emission_data[edge_id]['co2'].append(max(0, traci.edge.getCO2Emission(edge_id)))
                    emission_data[edge_id]['co'].append(max(0, traci.edge.getCOEmission(edge_id)))
                    emission_data[edge_id]['hc'].append(max(0, traci.edge.getHCEmission(edge_id)))
                    emission_data[edge_id]['nox'].append(max(0, traci.edge.getNOxEmission(edge_id)))
                    emission_data[edge_id]['pmx'].append(max(0, traci.edge.getPMxEmission(edge_id)))
                    emission_data[edge_id]['fuel'].append(max(0, traci.edge.getFuelConsumption(edge_id)))
                    emission_data[edge_id]['noise'].append(max(0, traci.edge.getNoiseEmission(edge_id)))

                    travel_times[edge_id].append(max(0, traci.edge.getTraveltime(edge_id)))
                    vehicle_counts[edge_id].append(traci.edge.getLastStepVehicleNumber(edge_id))
                    speeds[edge_id].append(traci.edge.getLastStepMeanSpeed(edge_id))

            # Output zone data at specified intervals
            if step > 0 and step % steps_per_output == 0:
                current_minute = step // 120  # Convert to minutes
                zone_data_frame = []

                for edge_id in edge_ids:
                    try:
                        edge = net.getEdge(edge_id)
                        zone = zone_dict.get(edge_id, 'unknown')

                        # Only process edges in A or B zones
                        if zone in ['A', 'B', 'border']:

                            # compute the average speed for the current interval.
                            recent_speeds = speeds[edge_id][-steps_per_output // 15:]
                            avg_interval_speed = np.mean(recent_speeds) if recent_speeds else edge.getSpeed()
                            free_flow_travel_time = edge.getLength() / avg_interval_speed if avg_interval_speed > 0 else np.inf

                            # Get raw travel times from recent steps
                            recent_tt = travel_times[edge_id][-steps_per_output // 15:]
                            raw_avg_tt = np.mean(recent_tt)
                            raw_max_tt = np.max(recent_tt)

                            # Define cap multipliers
                            cap_avg_tt = free_flow_travel_time * 10
                            cap_max_tt = free_flow_travel_time * 15

                            # Cap average and max travel times if needed
                            avg_travel_time = raw_avg_tt if raw_avg_tt <= cap_avg_tt else free_flow_travel_time
                            max_travel_time = raw_max_tt if raw_max_tt <= cap_max_tt else free_flow_travel_time

                            edge_info = {
                                'edge_id': edge_id,
                                'type': edge.getType(),
                                'length': edge.getLength(),
                                'from_junction': edge.getFromNode().getID(),
                                'to_junction': edge.getToNode().getID(),
                                'num_lanes': len(edge.getLanes()),
                                'speed_limit': edge.getSpeed(),
                                'priority': edge.getPriority(),
                                'has_traffic_light': 1 if (edge.getFromNode().getID() in traffic_lights or
                                                           edge.getToNode().getID() in traffic_lights) else 0,
                                'zone': zone,
                                'is_border': 1 if zone == 'border' else 0,
                                'connected_edges_other_zone': '',
                                'junction_complexity': len(edge.getFromNode().getOutgoing()) + len(
                                    edge.getToNode().getIncoming()),
                                'avg_vehicle_count': np.mean(vehicle_counts[edge_id][-steps_per_output // 15:]),
                                'max_vehicle_count': np.max(vehicle_counts[edge_id][-steps_per_output // 15:]),
                                'traffic_density': (np.mean(vehicle_counts[edge_id][-steps_per_output // 15:]) / edge.getLength()
                                                    if edge.getLength() > 0 else 0),
                                'avg_speed': avg_interval_speed,
                                'avg_travel_time': avg_travel_time,
                                'max_travel_time': max_travel_time,
                                'congestion_index': (
                                    avg_travel_time / free_flow_travel_time if free_flow_travel_time > 0 else 0
                                )
                            }

                            # Add emission metrics
                            for metric in ['co2', 'co', 'hc', 'nox', 'pmx', 'fuel', 'noise']:
                                metric_values = emission_data[edge_id][metric][-steps_per_output // 15:]
                                edge_info[f'avg_{metric}'] = np.mean(metric_values)
                                edge_info[f'max_{metric}'] = np.max(metric_values)
                                edge_info[f'{metric}_per_meter'] = (edge_info[f'avg_{metric}'] / edge.getLength()
                                                                    if edge.getLength() > 0 else 0)

                            # Identify connected edges from other zones
                            connected_edges = set()
                            for conn_edge in edge.getFromNode().getOutgoing() + edge.getFromNode().getIncoming():
                                conn_id = conn_edge.getID()
                                if conn_id not in edge_ids or conn_id == edge_id:
                                    continue
                                conn_zone = zone_dict.get(conn_id, 'unknown')
                                if (zone == 'A' and conn_zone in ['B', 'border']) or \
                                   (zone == 'B' and conn_zone in ['A', 'border']) or \
                                   (zone == 'border'):
                                    connected_edges.add(conn_id)

                            for conn_edge in edge.getToNode().getOutgoing() + edge.getToNode().getIncoming():
                                conn_id = conn_edge.getID()
                                if conn_id not in edge_ids or conn_id == edge_id:
                                    continue
                                conn_zone = zone_dict.get(conn_id, 'unknown')
                                if (zone == 'A' and conn_zone in ['B', 'border']) or \
                                   (zone == 'B' and conn_zone in ['A', 'border']) or \
                                   (zone == 'border'):
                                    connected_edges.add(conn_id)

                            edge_info['connected_edges_other_zone'] = ';'.join(connected_edges)

                            zone_data_frame.append(edge_info)
                    except Exception as e:
                        print(f"Warning: Skipping edge {edge_id} due to error: {str(e)}")
                        continue

                # Create and save zone-specific DataFrames
                df = pd.DataFrame(zone_data_frame)

                # Separate A and B zone data
                global_A, global_B = splitter(df)

                global_A_csv = f"{global_A_csv_prefix}{current_minute}min.csv"
                global_B_csv = f"{global_B_csv_prefix}{current_minute}min.csv"

                global_A.to_csv(global_A_csv, index=False)
                global_B.to_csv(global_B_csv, index=False)

                print(f"Zonal A data saved to {global_A_csv}")
                print(f"Zonal B data saved to {global_B_csv}")

                zone_data_history.extend(zone_data_frame)

        traci.close()
    except Exception as e:
        traci.close()
        raise Exception(f"Error during data extraction: {str(e)}")

    # Print and return the network boundary values
    print(f"\nBoundary values:\nmax_y: {max_y}\nmin_y: {min_y}\nmax_x: {max_x}\nmin_x: {min_x}\nsplit_x: {split_x}")
    return {
        'max_y': max_y,
        'min_y': min_y,
        'max_x': max_x,
        'min_x': min_x,
        'split_x': split_x
    }


# Example usage:
if __name__ == "__main__":
    net_file = "osm.net.xml.gz"
    sumo_cfg = "osm.sumocfg"
    results = process_simulation(net_file, sumo_cfg)
