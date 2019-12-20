#!/bin/sh

STATUS="ok"
GREP="$2"

for DB in $1
do
	echo "db.jobs.find({\"result.status\" : \"$STATUS\"}).toArray()" | mongo --host heika $DB
#| awk 'BEGIN {skip=1} /result/ {skip=0} /misc/ {skip=1} {if (! skip) {print}}'
done
