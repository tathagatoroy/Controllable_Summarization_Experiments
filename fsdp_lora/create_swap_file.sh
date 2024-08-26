#!/bin/bash

# Variables
SWAPFILE="$HOME/swapfile"
SWAPSIZE="20G"

# Check available disk space
echo "Checking available disk space..."
df -h "$HOME"

# Create the swap file
echo "Creating a $SWAPSIZE swap file at $SWAPFILE..."
fallocate -l $SWAPSIZE "$SWAPFILE" || { echo "fallocate failed, using dd..."; dd if=/dev/zero of="$SWAPFILE" bs=1G count=20; }

# Set the correct permissions
echo "Setting permissions for the swap file..."
chmod 600 "$SWAPFILE"

# Set up the swap space
echo "Setting up the swap space..."
mkswap "$SWAPFILE"

# Enable the swap file
echo "Enabling the swap file..."
swapon "$SWAPFILE"

# Verify the swap space
echo "Swap space enabled:"
swapon --show

# Making the swap permanent
echo "Adding swap file to /etc/fstab..."
echo "$SWAPFILE none swap sw 0 0" | tee -a /etc/fstab

# Adjusting swappiness (optional)
echo "Setting swappiness to 10..."
sysctl vm.swappiness=10

echo "Swap file creation and setup complete!"
