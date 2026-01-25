#!/bin/bash
# Manual GPU search with various filters for ClothGeom-NIF
# Usage: bash search_gpus.sh
#
# This script helps you explore available GPU options on Vast.ai
# before committing to creating an instance.

echo "🔍 Searching Vast.ai for ClothGeom-NIF compatible GPUs..."
echo "============================================================"
echo ""

# Check if vastai CLI is available
if ! command -v vastai &> /dev/null; then
    echo "❌ vastai CLI not found!"
    echo ""
    echo "Install with:"
    echo "   pip install vast"
    echo "   vastai set api-key YOUR_API_KEY"
    echo ""
    echo "Get your API key from: https://vast.ai/console/account/"
    exit 1
fi

echo "=== High-End Options (24GB+ VRAM) ==="
echo "Best for full training at 128³ resolution"
echo ""
vastai search offers \
  "rentable=true verified=true gpu_ram>=24 disk_space>=50" \
  --order dph_total --limit 10 2>/dev/null || echo "Search failed - check API key"

echo ""
echo "=== Budget Options (16-24GB VRAM) ==="
echo "May require reduced batch size or resolution"
echo ""
vastai search offers \
  "rentable=true verified=true gpu_ram>=16 gpu_ram<24 disk_space>=50" \
  --order dph_total --limit 10 2>/dev/null || echo "Search failed"

echo ""
echo "=== Specific GPU Models ==="
for GPU in "RTX_4090" "RTX_3090" "A6000" "RTX_A5000" "A100" "H100"; do
    echo ""
    echo "--- $GPU ---"
    vastai search offers \
      "gpu_name=$GPU rentable=true verified=true" \
      --order dph_total --limit 3 2>/dev/null | head -n 6 || echo "None available"
done

echo ""
echo "============================================================"
echo "💡 Usage Tips:"
echo ""
echo "To create an instance:"
echo "   vastai create instance <ID> --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime --disk 50 --ssh"
echo ""
echo "To see your instances:"
echo "   vastai show instances"
echo ""
echo "To destroy an instance:"
echo "   vastai destroy instance <ID>"
echo ""
echo "Recommended GPUs for ClothGeom-NIF:"
echo "   - RTX 4090 (24GB): Best performance/cost for training"
echo "   - RTX 3090 (24GB): Good budget option"
echo "   - A6000 (48GB): For larger batch sizes or higher resolution"
echo ""
echo "Expected costs:"
echo "   - RTX 3090: ~$0.20-0.40/hr"
echo "   - RTX 4090: ~$0.40-0.80/hr"
echo "   - A6000: ~$0.50-1.00/hr"
echo ""
echo "Full training (100 samples, 100 epochs): ~30-45 minutes = ~$0.50-1.00"
