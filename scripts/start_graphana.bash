#!/usr/bin/env bash

docker run -d -p 3000:3000 --restart=always --network=xilian grafana/grafana
