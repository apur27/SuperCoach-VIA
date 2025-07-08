import os
import shutil
from pathlib import Path
from PIL import Image, ImageOpt
import subprocess
import gzip
import json

class AssetOptimizer:
    """Optimize static assets for better performance."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.optimized_count = 0
        self.total_savings = 0

    def optimize_images(self, quality: int = 85, max_width: int = 1200):
        """Optimize PNG and other image files."""
        print("🖼️ Optimizing images...")
        
        image_extensions = {'.png', '.jpg', '.jpeg'}
        image_files = []
        
        # Find all image files
        for ext in image_extensions:
            image_files.extend(self.project_root.glob(f'**/*{ext}'))
        
        for image_path in image_files:
            if image_path.stat().st_size > 100 * 1024:  # Only optimize files > 100KB
                original_size = image_path.stat().st_size
                self.optimize_single_image(image_path, quality, max_width)
                new_size = image_path.stat().st_size
                savings = original_size - new_size
                
                if savings > 0:
                    self.total_savings += savings
                    self.optimized_count += 1
                    print(f"📉 Optimized {image_path.name}: {original_size//1024}KB → {new_size//1024}KB (saved {savings//1024}KB)")

    def optimize_single_image(self, image_path: Path, quality: int, max_width: int):
        """Optimize a single image file."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                # Resize if too large
                if img.width > max_width:
                    ratio = max_width / img.width
                    new_height = int(img.height * ratio)
                    img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
                
                # Save with optimization
                if image_path.suffix.lower() == '.png':
                    # For PNG, use compression
                    img.save(image_path, 'PNG', optimize=True, compress_level=6)
                else:
                    # For JPEG, use quality setting
                    img.save(image_path, 'JPEG', quality=quality, optimize=True)
                    
        except Exception as e:
            print(f"❌ Error optimizing {image_path}: {e}")

    def compress_large_files(self):
        """Compress large data files with gzip."""
        print("🗜️ Compressing large data files...")
        
        # Find large CSV files
        csv_files = list(self.project_root.glob('**/*.csv'))
        
        for csv_path in csv_files:
            if csv_path.stat().st_size > 1024 * 1024:  # Files > 1MB
                self.compress_file(csv_path)

    def compress_file(self, file_path: Path):
        """Compress a single file with gzip."""
        try:
            original_size = file_path.stat().st_size
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
            
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            compressed_size = compressed_path.stat().st_size
            compression_ratio = (1 - compressed_size / original_size) * 100
            
            print(f"🗜️ Compressed {file_path.name}: {original_size//1024}KB → {compressed_size//1024}KB ({compression_ratio:.1f}% reduction)")
            
            # Create decompression script
            self.create_decompression_script()
            
        except Exception as e:
            print(f"❌ Error compressing {file_path}: {e}")

    def create_decompression_script(self):
        """Create a script to decompress files when needed."""
        script_content = '''#!/usr/bin/env python3
"""Decompress data files when needed."""
import gzip
import shutil
from pathlib import Path

def decompress_data_files():
    """Decompress all .gz files in the project."""
    for gz_file in Path('.').glob('**/*.gz'):
        original_file = gz_file.with_suffix('')
        if not original_file.exists():
            print(f"Decompressing {gz_file}")
            with gzip.open(gz_file, 'rb') as f_in:
                with open(original_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

if __name__ == "__main__":
    decompress_data_files()
'''
        
        script_path = self.project_root / "decompress_data.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable on Unix systems
        try:
            os.chmod(script_path, 0o755)
        except:
            pass

    def optimize_model_files(self):
        """Optimize machine learning model files."""
        print("🤖 Optimizing model files...")
        
        model_dir = self.project_root / "models"
        if not model_dir.exists():
            return
        
        for model_file in model_dir.glob("*.joblib"):
            original_size = model_file.stat().st_size
            
            # Check if it's the large random forest model
            if "random_forest" in model_file.name and original_size > 50 * 1024 * 1024:
                print(f"⚠️ Large model detected: {model_file.name} ({original_size//1024//1024}MB)")
                self.optimize_large_model(model_file)

    def optimize_large_model(self, model_path: Path):
        """Optimize large model files by reducing precision or compression."""
        try:
            import joblib
            
            # Load the model
            model = joblib.load(model_path)
            
            # Create compressed version
            compressed_path = model_path.with_name(f"compressed_{model_path.name}")
            joblib.dump(model, compressed_path, compress=('lz4', 3))
            
            original_size = model_path.stat().st_size
            compressed_size = compressed_path.stat().st_size
            savings = original_size - compressed_size
            
            if savings > 0:
                print(f"📦 Compressed model: {original_size//1024//1024}MB → {compressed_size//1024//1024}MB")
                
                # Replace original with compressed version
                model_path.unlink()
                compressed_path.rename(model_path)
                
                self.total_savings += savings
            else:
                compressed_path.unlink()
                
        except Exception as e:
            print(f"❌ Error optimizing model {model_path}: {e}")

    def clean_cache_files(self):
        """Clean up cache and temporary files."""
        print("🧹 Cleaning cache files...")
        
        cache_patterns = [
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pyo",
            "**/.pytest_cache",
            "**/cache",
            "**/.DS_Store",
            "**/Thumbs.db"
        ]
        
        removed_count = 0
        
        for pattern in cache_patterns:
            for item in self.project_root.glob(pattern):
                try:
                    if item.is_file():
                        item.unlink()
                        removed_count += 1
                    elif item.is_dir():
                        shutil.rmtree(item)
                        removed_count += 1
                except Exception as e:
                    print(f"⚠️ Could not remove {item}: {e}")
        
        if removed_count > 0:
            print(f"🗑️ Removed {removed_count} cache files/directories")

    def create_optimization_report(self):
        """Create a report of optimizations performed."""
        report = {
            "optimization_summary": {
                "files_optimized": self.optimized_count,
                "total_savings_mb": round(self.total_savings / (1024 * 1024), 2),
                "optimizations_performed": [
                    "Image compression and resizing",
                    "Large file compression",
                    "Model file optimization",
                    "Cache cleanup"
                ]
            },
            "recommendations": [
                "Use optimized_prediction.py instead of prediction.py/prediction_cpu.py",
                "Use optimized_scraper.py for web scraping",
                "Enable caching for repeated operations",
                "Consider removing unused dependencies",
                "Use compressed data files in production"
            ]
        }
        
        report_path = self.project_root / "optimization_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Optimization report saved to {report_path}")

    def run_full_optimization(self):
        """Run complete asset optimization pipeline."""
        print("🚀 Starting Full Asset Optimization...")
        
        # Run all optimizations
        self.optimize_images(quality=85, max_width=1200)
        self.compress_large_files()
        self.optimize_model_files()
        self.clean_cache_files()
        
        # Create report
        self.create_optimization_report()
        
        print(f"\n✅ Optimization completed!")
        print(f"📈 Total space saved: {self.total_savings // 1024 // 1024}MB")
        print(f"📁 Files optimized: {self.optimized_count}")

if __name__ == "__main__":
    optimizer = AssetOptimizer()
    optimizer.run_full_optimization()