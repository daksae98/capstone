import csv
f = open('write.csv','w', newline='')
wr = csv.writer(f)
wr.writerow([1,'림코딩', '부산'])
wr.writerow([2,'김갑환', '서울'])
 
f.close()