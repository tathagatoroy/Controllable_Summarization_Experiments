#!/bin/bash 

HUGGINGFACE_TOKEN="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"

# Perform the login using the token
# TEMP_FILE=$(mktemp)

# # Write the token to the temporary file
# echo "$HUGGINGFACE_TOKEN" > "$TEMP_FILE"

# # Perform the login using the token from the temporary file
# huggingface-cli login < "$TEMP_FILE"

huggingface-cli login --token $HUGGINGFACE_TOKEN
