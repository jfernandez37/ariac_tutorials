import yaml
import math

def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> list:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = [0] * 4
    
    q[0] = cy * cp * sr - sy * sp * cr
    q[1] = sy * cp * sr + cy * sp * cr
    q[2] = sy * cp * cr - cy * sp * sr
    q[3] = cy * cp * cr + sy * sp * sr
    q = [round(val,4) for val in q]
    return q

objects_dict = {}

# BINS
bin_positions = [[x,y,0.0] 
                 for x in [-1.9,-2.65]
                 for y in [3.375,2.625,-3.375,-2.625]]
bin_rotations = [quaternion_from_euler(0,0,math.pi) for _ in range(8)]
for i in range(1,9):
    objects_dict[f"bin{i}"]={"position":bin_positions[i-1],"orientation":bin_rotations[i-1],"file":"bin.stl"}

# ASSEMBLY STATIONS
assembly_stations_positions = [[x,y,0.0]
                               for x in [-7.3,-12.3]
                               for y in [3.0,-3.0]]
assembly_stations_rotations = [quaternion_from_euler(0,0,0) for _ in range(4)]

for i in range(1,5):
    objects_dict[f"as{i}"] = {"position":assembly_stations_positions[i-1],"orientation":assembly_stations_rotations[i-1],"file":"assembly_station.stl"}


# ASSEMBLY BRIEFCASES
assembly_insert_positions = [[x,y,1.011]
                                for x in [-7.7,-12.7]
                                for y in [3.0,-3.0]]

for i in range(1,5):
    objects_dict[f"assembly_insert{i}"] = {"position":assembly_insert_positions[i-1],"orientation":quaternion_from_euler(0,0,0),"file":"assembly_insert.stl"}

# CONVEYOR BELT
objects_dict["conveyor"] = {"position":[-0.6,0,0], "orientation":quaternion_from_euler(0,0,0), "file":"conveyor.stl"}

# KITTING TABLES
kitting_table_positions = [[-1.3,y,0.0] for y in [5.84,-5.84]]
kitting_table_rotations = [quaternion_from_euler(0,0,0),quaternion_from_euler(0,0,math.pi)]

objects_dict["kts1_table"] = {"position":[-1.3,5.84,0.0],"orientation":quaternion_from_euler(0,0,0),"file":"kit_tray_table.stl"}
objects_dict["kts2_table"] = {"position":[-1.3,-5.84,0.0],"orientation":quaternion_from_euler(0,0,math.pi),"file":"kit_tray_table.stl"}
print("The python dictionary is:")
print(objects_dict)
file=open("collision_object_info.yaml","w")
yaml.dump(objects_dict,file, sort_keys=False)
file.close()
print("YAML file saved.")