#!/bin/bash
# Fix broken symlinks in indoor6_ace datasets
# All rgb symlinks point to /data/xwh but should point to /mnt/storage/xwh

set -e

ACE_ROOT="/mnt/storage/xwh/indoor6_ace"
IMG_ROOT="/mnt/storage/xwh/indoor6"

echo "=== Fixing Indoor6 ACE Dataset Symlinks ==="
echo ""

# Find all scenes
for scene_dir in "$ACE_ROOT"/scene*; do
    if [ ! -d "$scene_dir" ]; then
        continue
    fi

    scene_name=$(basename "$scene_dir")
    echo "Processing $scene_name..."

    # Process train and test splits
    for split in train test; do
        rgb_dir="$scene_dir/$split/rgb"

        if [ ! -d "$rgb_dir" ]; then
            echo "  [SKIP] $split/rgb not found"
            continue
        fi

        # Check if symlinks are broken
        first_file="$rgb_dir/000000.jpg"
        if [ -L "$first_file" ] && [ ! -e "$first_file" ]; then
            echo "  [FIX] $split/rgb has broken symlinks"

            # Source images directory
            img_dir="$IMG_ROOT/$scene_name/images"

            if [ ! -d "$img_dir" ]; then
                echo "  [ERROR] Source images not found: $img_dir"
                continue
            fi

            # Remove old symlinks
            rm -f "$rgb_dir"/*.jpg

            # Create new symlinks based on pose files (ensures count match)
            count=0
            for pose in "$scene_dir/$split/poses"/*.pose; do
                if [ ! -f "$pose" ]; then
                    continue
                fi
                num=$(basename "$pose" .pose)
                src="$img_dir/image-${num}.color.jpg"
                if [ -f "$src" ]; then
                    ln -s "$src" "$rgb_dir/${num}.jpg"
                    count=$((count + 1))
                fi
            done

            echo "  [OK] Created $count symlinks in $split/rgb"
        else
            echo "  [OK] $split/rgb symlinks are valid"
        fi
    done
    echo ""
done

echo "=== Done ==="
