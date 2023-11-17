import yaml

objects_dict = {}

# BINS
bin_positions = [[x,y,0.0] 
                 for x in [-1.9,-2.65]
                 for y in [3.375,2.625,-3.375,-2.625]]
bin_rotations = [[0.0,0.0,0.0,1.0] for _ in range(8)]
for i in range(1,9):
    objects_dict[f"bin{i}"]={"position":bin_positions[i-1],"orientation":bin_rotations[i-1],"file":"bin.stl"}

# ASSEMBLY STATIONS
assembly_stations_positions = [[x,y,0.0]
                               for x in [-7.3,-12.3]
                               for y in [3,-3]]
assembly_stations_rotations = [[0.0,0.0,0.0,1.0] for _ in range(4)]

for i in range(1,5):
    objects_dict[f"as{i}"] = {"position":assembly_stations_positions[i-1],"orientation":assembly_stations_rotations[i-1],"file":"assembly_station.stl"}

# ASSEMBLY BRIEFCASES
assembly_briefcase_positions = [[x,y,0.0]
                                for x in [-7.7,-12.7]
                                for y in [3,-3]]
assembly_briefcase_rotations = [[0.0,0.0,0.0,1.0] for _ in range(4)]

for i in range(1,5):
    objects_dict[f"assembly_insert{i}"] = {"position":assembly_briefcase_positions[i-1],"orientation":assembly_briefcase_rotations[i-1],"file":"assembly_insert.stl"}

# CONVEYOR BELT
objects_dict["conveyor"] = {"position":[-0.6,0,0], "orientation":[0.0,0.0,0.0,1.0], "file":"conveyor.stl"}

# KITTING TABLES
kitting_table_positions = [[-1.3,y,0.0] for y in [5.84,-5.84]]
kitting_table_rotations = [[0.0,0.0,0.0,1.0] for _ in range(2)]

for i in range(1,3):
    objects_dict[f"kts{i}_table"] = {"position":kitting_table_positions[i-1],"orientation":kitting_table_rotations[i-1],"file":"kit_tray_table.stl"}

print("The python dictionary is:")
print(objects_dict)
file=open("collision_object_info.yaml","w")
yaml.dump(objects_dict,file, sort_keys=False)
file.close()
print("YAML file saved.")