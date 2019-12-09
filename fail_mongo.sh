#!/bin/sh

echo 'db.jobs.find({"result.status" : "fail"}).toArray()' | mongo --host heika plos1-m3-007g | egrep -A 1 exception
