#!/bin/bash

cd spice-coco-caption
python myeval.py ../coco-caption/$1 $2
cd ../
