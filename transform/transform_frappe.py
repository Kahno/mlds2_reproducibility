def process_line(line):
    lines = line.strip().split()
    result = lines[0]
    features = lines[1:]
    feats = ",".join([f.split(":")[0] for f in features])    
    return f"{result},{feats}"
    

header = "click," + ",".join([f"c{i+1}" for i in range(10)])    

file_counter = 0
line_counter = 0
g = open(f"frappe_data/test_{file_counter}.csv", "w")
print(f"Writing file number {file_counter}")
g.write(header + "\n")

with open("test.libsvm", "r") as f:
    for line in f:
        if line_counter == 1500000:
            g.close()
            file_counter += 1
            g = open(f"frappe_data/test_{file_counter}.csv", "w")
            print(f"Writing file number {file_counter}")
            g.write(header + "\n")
            line_counter = 0
            
        g.write(process_line(line) + "\n")
        line_counter += 1
        
g.close()