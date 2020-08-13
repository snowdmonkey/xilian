#!/usr/bin/env bash

docker network create xilian

docker run -d \
  --name=influxdb \
  -p 8086:8086 \
  --restart=always \
  -e INFLUXDB_DB=xilian \
  -e INFLUXDB_ADMIN_USER=admin -e INFLUXDB_ADMIN_PASSWORD=2much4ME \
  --network=xilian \
  --hostname=influx \
  influxdb:1.8.1

