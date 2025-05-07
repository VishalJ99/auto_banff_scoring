#!/bin/bash

# Extract lines containing 'pas' (case insensitive) from stain_classified_cases.txt
grep -i "pas" data/stain_classified_cases.txt > pas_cases.txt

echo "PAS cases have been extracted to pas_cases.txt" 