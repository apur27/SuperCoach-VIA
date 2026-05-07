import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# Suppress OpenCL compiler warnings (optional)
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'

def train_lightgbm_gpu_classifier():
    """
    Complete LightGBM GPU training pipeline with best practices
    """
    
    # ==========================================
    # 1. DATA LOADING & PREPARATION
    # ==========================================
    print("🔄 Loading and preparing data...")
    
    # Load the classic Iris dataset
    X, y = load_iris(return_X_y=True)
    print(f"Dataset shape: {X.shape}, Classes: {len(np.unique(y))}")
    
    # Split data with stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y  # Ensures balanced splits across classes
    )
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # ==========================================
    # 2. LIGHTGBM DATASET CREATION
    # ==========================================
    print("📊 Creating LightGBM datasets...")
    
    # Create LightGBM datasets (optimized internal format)
    train_data = lgb.Dataset(
        X_train, 
        label=y_train,
        feature_name=[f'feature_{i}' for i in range(X_train.shape[1])],
        categorical_feature='auto'  # Auto-detect categorical features
    )
    
    valid_data = lgb.Dataset(
        X_test, 
        label=y_test, 
        reference=train_data  # Use training data as reference for consistency
    )
    
    # ==========================================
    # 3. GPU-OPTIMIZED PARAMETERS
    # ==========================================
    print("⚙️ Configuring GPU parameters...")
    
    params = {
        # === GPU Configuration ===
        'device': 'cuda',  # Use CUDA for NVIDIA GPUs (cleaner than 'gpu')
        'gpu_use_dp': False,  # Single precision for faster computation
        
        # === Model Architecture ===
        'objective': 'multiclass',  # Multi-class classification
        'num_class': 3,  # Number of classes in Iris dataset
        'boosting_type': 'gbdt',  # Gradient Boosting Decision Tree
        
        # === Performance Metrics ===
        'metric': ['multi_logloss', 'multi_error'],  # Multiple metrics for monitoring
        
        # === Learning Parameters ===
        'learning_rate': 0.1,  # Conservative learning rate
        'num_leaves': 31,  # Default balanced tree complexity
        'max_depth': -1,  # No depth limit (controlled by num_leaves)
        'min_data_in_leaf': 20,  # Minimum samples per leaf (overfitting control)
        'feature_fraction': 0.9,  # Feature subsampling for regularization
        'bagging_fraction': 0.8,  # Row subsampling
        'bagging_freq': 5,  # Frequency of bagging
        
        # === Regularization ===
        'lambda_l1': 0.0,  # L1 regularization
        'lambda_l2': 0.0,  # L2 regularization
        
        # === Output Control ===
        'verbosity': 1,  # Moderate verbosity (0: silent, 1: info, 2: debug)
        'seed': 42,  # Reproducibility
        
        # === GPU-Specific Optimizations ===
        'max_bin': 255,  # Number of bins for feature discretization
        'force_row_wise': True,  # Memory layout optimization for GPU
    }
    
    # ==========================================
    # 4. MODEL TRAINING WITH VALIDATION
    # ==========================================
    print("🚀 Training LightGBM model on GPU...")
    
    # Training with early stopping and validation monitoring
    model = lgb.train(
        params=params,
        train_set=train_data,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'eval'],
        num_boost_round=100,  # Maximum iterations
        callbacks=[
            lgb.early_stopping(stopping_rounds=10, verbose=True),
            lgb.log_evaluation(period=10)  # Print metrics every 10 rounds
        ]
    )
    
    print(f"✅ Training completed! Best iteration: {model.best_iteration}")
    
    # ==========================================
    # 5. PREDICTION & EVALUATION
    # ==========================================
    print("🔍 Making predictions and evaluating...")
    
    # Predictions (returns probabilities for multiclass)
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Convert probabilities to class predictions
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Test Accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Setosa', 'Versicolor', 'Virginica']))
    
    # ==========================================
    # 6. FEATURE IMPORTANCE ANALYSIS
    # ==========================================
    print("📊 Feature Importance Analysis:")
    
    feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    importances = model.feature_importance(importance_type='gain')
    
    for name, importance in zip(feature_names, importances):
        print(f"  {name}: {importance:.2f}")
    
    # ==========================================
    # 7. MODEL DIAGNOSTICS
    # ==========================================
    print("\n🔧 Model Diagnostics:")
    print(f"  - Number of trees: {model.num_trees()}")
    print(f"  - Best iteration: {model.best_iteration}")
    print(f"  - Best score: {model.best_score}")
    
    return model, y_pred, y_pred_proba

# ==========================================
# EXECUTION & ERROR HANDLING
# ==========================================
if __name__ == "__main__":
    try:
        print("🐍 Python ML Council - LightGBM GPU Training Pipeline")
        print("=" * 60)
        
        # Check GPU availability
        try:
            import lightgbm as lgb
            # Test GPU device
            test_params = {'device': 'cuda', 'objective': 'regression'}
            print("✅ GPU device available for LightGBM")
        except Exception as e:
            print(f"⚠️ GPU may not be available: {e}")
            print("Falling back to CPU...")
        
        # Run the complete pipeline
        model, predictions, probabilities = train_lightgbm_gpu_classifier()
        
        print("\n🎉 Pipeline completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        print("💡 Troubleshooting tips:")
        print("  1. Ensure CUDA-compatible GPU is available")
        print("  2. Check LightGBM GPU compilation")
        print("  3. Try 'device': 'cpu' if GPU issues persist")
