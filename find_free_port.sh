#!/bin/bash

let DISPLAY_PORT=8097

while [[ -n `netstat -antu | grep $DISPLAY_PORT` ]]; do
	let DISPLAY_PORT=$DISPLAY_PORT+1
done