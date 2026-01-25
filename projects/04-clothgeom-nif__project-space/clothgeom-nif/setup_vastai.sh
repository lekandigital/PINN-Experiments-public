#!/bin/bash
# Automated Vast.ai GPU instance setup for ClothGeom-NIF
# Usage: bash setup_vastai.sh
#
# Prerequisites:
#   pip install vast
#   vastai set api-key YOUR_API_KEY

set -e  # Exit on any error

echo "🔍 Searching for suitable GPU instances..."
echo ""

# Create temp directory for search results
mkdir -p .vastai_search

# Search for available GPUs (in order of preference)
# Priority: RTX 4090 > RTX 3090 > A6000 > RTX A5000
echo "Searching RTX 4090..."
vastai search offers \
  "gpu_name=RTX_4090 rentable=true verified=true gpu_ram>=24 disk_space>=50" \
  --order dph_total --limit 5 2>/dev/null > .vastai_search/gpu_search_4090.txt || true

echo "Searching RTX 3090..."
vastai search offers \
  "gpu_name=RTX_3090 rentable=true verified=true gpu_ram>=24 disk_space>=50" \
  --order dph_total --limit 5 2>/dev/null > .vastai_search/gpu_search_3090.txt || true

echo "Searching A6000..."
vastai search offers \
  "gpu_name=A6000 rentable=true verified=true gpu_ram>=24 disk_space>=50" \
  --order dph_total --limit 5 2>/dev/null > .vastai_search/gpu_search_a6000.txt || true

echo "Searching RTX A5000..."
vastai search offers \
  "gpu_name=RTX_A5000 rentable=true verified=true gpu_ram>=24 disk_space>=50" \
  --order dph_total --limit 5 2>/dev/null > .vastai_search/gpu_search_a5000.txt || true

# Display results
echo ""
echo "📊 Available instances:"
echo "========================"
echo ""
echo "--- RTX 4090 ---"
cat .vastai_search/gpu_search_4090.txt 2>/dev/null || echo "No results"
echo ""
echo "--- RTX 3090 ---"
cat .vastai_search/gpu_search_3090.txt 2>/dev/null || echo "No results"
echo ""
echo "--- A6000 ---"
cat .vastai_search/gpu_search_a6000.txt 2>/dev/null || echo "No results"
echo ""
echo "--- RTX A5000 ---"
cat .vastai_search/gpu_search_a5000.txt 2>/dev/null || echo "No results"

# Parse the cheapest instance ID (from first non-header line with valid data)
INSTANCE_ID=""
for file in .vastai_search/gpu_search_*.txt; do
    if [ -f "$file" ]; then
        # Get first line that starts with a number (instance ID)
        ID=$(grep -E "^[0-9]+" "$file" 2>/dev/null | head -n1 | awk '{print $1}')
        if [ -n "$ID" ]; then
            INSTANCE_ID=$ID
            break
        fi
    fi
done

if [ -z "$INSTANCE_ID" ]; then
    echo ""
    echo "❌ No suitable instances found with specific GPU filters."
    echo ""
    echo "💡 Trying broader search (any GPU with 20GB+ VRAM)..."
    vastai search offers \
      "rentable=true verified=true gpu_ram>=20 disk_space>=50" \
      --order dph_total --limit 10 > .vastai_search/gpu_search_any.txt 2>/dev/null || true
    
    cat .vastai_search/gpu_search_any.txt
    
    INSTANCE_ID=$(grep -E "^[0-9]+" .vastai_search/gpu_search_any.txt 2>/dev/null | head -n1 | awk '{print $1}')
    
    if [ -z "$INSTANCE_ID" ]; then
        echo ""
        echo "❌ Still no instances found. Possible reasons:"
        echo "   - All GPUs are currently rented"
        echo "   - Network/API issues"
        echo "   - Invalid API key"
        echo ""
        echo "Try manual search:"
        echo "   vastai search offers 'rentable=true' --order dph_total --limit 20"
        exit 1
    fi
fi

echo ""
echo "✅ Selected instance ID: $INSTANCE_ID"
echo ""

# Ask for confirmation before creating
read -p "Create this instance? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Create instance with PyTorch image
echo "🚀 Creating instance..."
vastai create instance $INSTANCE_ID \
  --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
  --disk 50 \
  --ssh

echo ""
echo "⏳ Waiting for instance to start (this may take 1-2 minutes)..."
sleep 30

# Get instance details
echo ""
echo "📋 Instance details:"
vastai show instances

# Save instance info for later use
vastai show instances > .vastai_search/current_instance.txt

echo ""
echo "✅ Instance created!"
echo ""
echo "📝 Next steps:"
echo "   1. Wait ~1 minute for SSH to become available"
echo "   2. Run: bash connect_instance.sh"
echo "   3. Or manually: ssh root@<host> -p <port>"
echo ""
echo "🛑 To destroy instance when done:"
echo "   vastai destroy instance <instance_id>"
echo ""

# Clean up temp files
# rm -rf .vastai_search
