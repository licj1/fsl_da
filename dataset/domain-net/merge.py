ft = open('oldtrain.txt').readlines()
fv = open('val_new_domain.txt').readlines()
fw = open('train.txt','w')

for line in ft:
    line = line.strip()
    print(line, file=fw)

for line in fv:
    line = line.strip().split()
    print(line[0], str(int(line[1])+220), file=fw)
for line in fv:
    line = line.strip().split()
    print(line[0], str(int(line[1])+220), file=fw)
for line in fv:
    line = line.strip().split()
    print(line[0], str(int(line[1])+220), file=fw)
