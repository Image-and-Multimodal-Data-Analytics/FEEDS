#!/bin/bash

CSV_FILE="cases_to_generate_pseudo_labels_for_3.csv"
SOURCE_DIR="/dartfs/rc/lab/B/BhattacharyaI/Public_Datasets/Autopet_III_nnunet_raw/Dataset888_AutoPet/imagesTr"
REMOTE="unity:/scratch4/workspace/f007g3j_dartmouth_edu-simple/nnUNet_data/nnUNet_results/train_80_pl"

tail -n +2 "$CSV_FILE" | sed 's/\r//' | while IFS= read -r line; do
    [[ -z "$line" ]] && continue

    # Trim only leading/trailing whitespace, preserve internal spaces
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"

    echo "Syncing: $line"

    rsync -avzL --progress --ignore-existing "${SOURCE_DIR}/${line}_0000.nii.gz" "${REMOTE}/"
    rsync -avzL --progress --ignore-existing "${SOURCE_DIR}/${line}_0001.nii.gz" "${REMOTE}/"

done

echo "Done!"