#! /usr/bin/env bash

for filename in $1/*.png; do
name=${filename##*/};
base=${name%.png};
echo "$1/$name"
convert "$1/$name" "$2/${name}" +append "$2/${base}_merged.png" 
done


