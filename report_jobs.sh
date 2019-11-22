#!/bin/sh

if [ $# -eq 0 ]
then
	STATUS="ok"
else
	STATUS=$1
fi

echo "sMAPE,model.type,experiment,start.time,end.time"
#for DB in plos1-m3-001g plos1-m3-002g plos1-m3-003g
for DB in plos1-m3-005g
do
	#echo "db.jobs.find({\"result.status\" : \"$STATUS\"}).toArray()" | mongo --host heika $DB | awk '/loss/ {printf("%8.4f,", $NF)} /exp_key/ {printf("%s", $NF)} /book_time/ {printf("%10s,", substr($NF, 10, 24))} /refresh_time/ {print substr($NF, 10, 24)}'
	echo "db.jobs.find({\"result.status\" : \"$STATUS\"}).toArray()" | mongo --host heika $DB | awk '/loss/ {printf("%8.4f,", $NF)} /exp_key/ {printf("%s", $NF)} /"type"/ {printf("%10s,", $NF)} /book_time/ {printf("%10s", $NF)} /refresh_time/ {print $NF}'
done | grep -v "^nan" | sed 's/ISODate//g' | tr -d "()"
