#!/bin/sh

if [ $# -eq 0 ]
then
	STATUS="ok"
else
	STATUS=$1
fi

echo "sMAPE,experiment,start.date,end.date"
for DB in plos1-m3-001g plos1-m3-002g
do
	echo "db.jobs.find({\"result.status\" : \"$STATUS\"}).toArray()" | mongo --host heika $DB | awk '/loss/ {printf("%.4f,", $NF)} /exp_key/ {printf("%s", $NF)} /book_time/ {printf("%10s,", substr($NF, 10, 24))} /refresh_time/ {print substr($NF, 10, 24)}'
done | grep -v "^nan"
