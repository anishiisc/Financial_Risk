import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import xgboost as xgb
from lightgbm import LGBMClassifier
import warnings
import os
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set the global font sizes and styles for all plots
plt.rcParams['font.size'] = 14  # Increase base font size
plt.rcParams['font.weight'] = 'bold'  # Make fonts bold by default
plt.rcParams['axes.titlesize'] = 18  # Larger title
plt.rcParams['axes.titleweight'] = 'bold'  # Bold title
plt.rcParams['axes.labelsize'] = 16  # Larger axis labels
plt.rcParams['axes.labelweight'] = 'bold'  # Bold axis labels
plt.rcParams['xtick.labelsize'] = 14  # Larger tick labels
plt.rcParams['ytick.labelsize'] = 14  # Larger tick labels
plt.rcParams['legend.fontsize'] = 14  # Larger legend
plt.rcParams['figure.titlesize'] = 20  # Larger figure title
plt.rcParams['figure.titleweight'] = 'bold'  # Bold figure title

# Create directories for saving results
os.makedirs('plots', exist_ok=True)
os.makedirs('reports', exist_ok=True)
os.makedirs('sampling_reports', exist_ok=True)
os.makedirs('sampling_distributions', exist_ok=True)

# 1. Generate dummy credit rating data
def generate_credit_rating_data(n_samples=10000):
    """
    Generate dummy credit rating data with fewer classes and more balanced distribution.
    The first 4 features are more important than the others.
    
    Modified to ensure at least 10 samples per class to avoid issues with ADASYN and SMOTE.
    """
    # Create 10 features
    X = np.random.randn(n_samples, 10)
    
    # Make the first 4 features more important by scaling them
    X[:, 0] *= 5  # Income to debt ratio (higher is better)
    X[:, 1] *= 4  # Payment history (higher is better)
    X[:, 2] *= 3  # Credit utilization (lower is better)
    X[:, 3] *= 2  # Length of credit history (higher is better)
    
    # Create score based on important features
    score = 3*X[:, 0] - 2*X[:, 2] + 1.5*X[:, 1] + X[:, 3] + 0.2*np.sum(X[:, 4:], axis=1)
    
    # Normalize score to be between 0 and 1
    score = (score - np.min(score)) / (np.max(score) - np.min(score))
    
    # Apply a better distribution to create a more balanced set
    # Using a modified approach that results in more samples in small classes
    score = np.random.beta(2, 2, n_samples) * 0.7 + score * 0.3  # More balanced beta distribution
    
    # Scale to 0-7 to have fewer classes with more samples in each
    y = np.floor(score * 7.99).astype(int)
    
    # Create feature names
    feature_names = [
        'income_to_debt_ratio',
        'payment_history_score',
        'credit_utilization',
        'credit_history_length',
        'num_credit_accounts',
        'num_recent_inquiries',
        'num_delinquencies',
        'total_credit_limit',
        'revolving_balance',
        'installment_balance'
    ]
    
    # Create class names (reduced from 16 to 8)
    class_names = [
        'AAA', 'AA', 'A+', 'A', 
        'BBB+', 'BBB', 'BB+', 'BB'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['credit_rating'] = y
    df['credit_rating_label'] = [class_names[i] for i in y]
    
    # Check if any class has less than 10 samples
    class_counts = df['credit_rating'].value_counts()
    print("Initial class distribution:")
    for class_idx, count in class_counts.items():
        print(f"Class {class_idx} ({class_names[class_idx]}): {count} samples")
    
    # Find classes with less than 10 samples
    rare_classes = class_counts[class_counts < 10].index.tolist()
    
    # If rare classes exist, add more samples to them directly
    if rare_classes:
        print(f"Found {len(rare_classes)} classes with less than 10 samples, adding more samples...")
        for rare_class in rare_classes:
            # Calculate how many more samples we need
            samples_needed = 10 - class_counts[rare_class]
            
            # Create synthetic samples for this class
            # Get mean and std of existing samples for this class
            class_samples = df[df['credit_rating'] == rare_class].drop(['credit_rating', 'credit_rating_label'], axis=1)
            
            if len(class_samples) > 0:  # Avoid division by zero or empty arrays
                mean_vector = class_samples.mean().values
                std_vector = class_samples.std().values + 0.01  # Add small epsilon to avoid zero std
                
                # Generate new samples around the mean with similar std
                new_samples = np.random.normal(loc=mean_vector, scale=std_vector, size=(samples_needed, len(mean_vector)))
                
                # Create DataFrame for new samples
                new_df = pd.DataFrame(new_samples, columns=feature_names)
                new_df['credit_rating'] = rare_class
                new_df['credit_rating_label'] = class_names[rare_class]
                
                # Append to original DataFrame
                df = pd.concat([df, new_df], ignore_index=True)
    
    # Verify final class counts
    final_class_counts = df['credit_rating'].value_counts()
    print("\nFinal class distribution:")
    for class_idx, count in final_class_counts.items():
        print(f"Class {class_idx} ({class_names[class_idx]}): {count} samples")
    
    return df, class_names

# 2. Analyze class distribution
def analyze_class_distribution(df, class_names):
    """
    Analyze and visualize the class distribution to identify imbalance.
    """
    plt.figure(figsize=(16, 8))  # Increased figure size for better visibility
    class_counts = df['credit_rating'].value_counts().sort_index()
    
    bars = plt.bar(range(len(class_counts)), [class_counts.get(i, 0) for i in range(len(class_counts))])
    
    # Only show labels for classes that actually exist in the data
    tick_positions = range(len(class_counts))
    tick_labels = [class_names[i] if i < len(class_names) else f"Class {i}" for i in class_counts.index]
    
    plt.xticks(tick_positions, tick_labels, rotation=45, ha='right')  # Changed rotation for better readability
    plt.title('Class Distribution', fontsize=22, fontweight='bold', pad=20)  # Extra padding for title
    plt.xlabel('Credit Rating Class', fontsize=18, fontweight='bold', labelpad=15)  # Extra padding for label
    plt.ylabel('Count', fontsize=18, fontweight='bold', labelpad=15)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 50,
                 f'{height}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the figure with high resolution
    plt.savefig('plots/class_distribution.png', dpi=300, bbox_inches='tight')
    
    # Calculate imbalance metrics
    total_samples = len(df)
    min_class_count = class_counts.min()
    max_class_count = class_counts.max()
    imbalance_ratio = max_class_count / min_class_count
    
    print(f"Total samples: {total_samples}")
    print(f"Minimum class count: {min_class_count}")
    print(f"Maximum class count: {max_class_count}")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")
    
    return class_counts

# 3. Feature importance analysis
def analyze_feature_importance(df, X_train, y_train, feature_names):
    """
    Analyze feature importance using Random Forest.
    """
    # Train a Random Forest to get feature importances
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Plot feature importances
    plt.figure(figsize=(14, 8))  # Increased figure size
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    bars = plt.bar(range(X_train.shape[1]), importances[indices])
    plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.title('Feature Importances', fontsize=22, fontweight='bold', pad=20)
    plt.ylabel('Importance Score', fontsize=18, fontweight='bold', labelpad=15)
    plt.xlabel('Features', fontsize=18, fontweight='bold', labelpad=15)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{height:.3f}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the figure with high resolution
    plt.savefig('plots/feature_importances.png', dpi=300, bbox_inches='tight')
    
    # Print feature importances
    print("Feature importances:")
    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")
    
    # Save feature importances to a formatted text file
    with open('reports/feature_importances.txt', 'w') as f:
        f.write("Feature Importances\n")
        f.write("===================\n\n")
        for i in indices:
            f.write(f"{feature_names[i]}: {importances[i]:.4f}\n")
    
    # Calculate correlation matrix between features
    correlation_matrix = df[feature_names].corr()
    plt.figure(figsize=(14, 12))  # Larger figure
    
    # Create heatmap with improved readability
    heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                 linewidths=0.5, annot_kws={"size": 12, "weight": "bold"})
    
    # Improve heatmap labels
    plt.title('Feature Correlation Matrix', fontsize=22, fontweight='bold', pad=20)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    
    # Save the correlation matrix figure with high resolution
    plt.savefig('plots/feature_correlation_matrix.png', dpi=300, bbox_inches='tight')
    
    return importances, indices

# 4. Evaluate models before sampling
def evaluate_models(X_train, y_train, X_test, y_test, class_names, reverse_mapping):
    """
    Evaluate multiple models on the original data.
    """
    # Since we've applied a class mapping, we need to adjust class_names accordingly
    present_class_names = []
    for i in range(len(np.unique(np.concatenate([y_train, y_test])))):
        original_class = reverse_mapping[i]
        if original_class < len(class_names):
            present_class_names.append(class_names[original_class])
        else:
            present_class_names.append(f"Class {original_class}")
    
    # Find the unique classes in the training data
    unique_train_classes = np.unique(y_train)
    print(f"Unique training classes: {unique_train_classes}")
    
    # Create a directory for classification reports if it doesn't exist
    import os
    if not os.path.exists('reports'):
        os.makedirs('reports')
    
    models = {
        'Logistic Regression': LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, C=0.1),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, enable_categorical=False),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, C=0.1, gamma='scale'),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        model_name_safe = name.replace(' ', '_')
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        # Print accuracy
        print(f"{name} accuracy: {accuracy:.4f}")
        
        # Format and save classification report
        report = classification_report(y_test, y_pred, target_names=present_class_names, zero_division=0, output_dict=True)
        
        # Print prettier classification report
        print("\nClassification Report:")
        print("-" * 70)
        print(f"{'Class':<15}{'Precision':<15}{'Recall':<15}{'F1-score':<15}{'Support':<10}")
        print("-" * 70)
        
        for i, class_name in enumerate(present_class_names):
            if class_name in report:
                cls_data = report[class_name]
                print(f"{class_name:<15}{cls_data['precision']:.4f}       {cls_data['recall']:.4f}       {cls_data['f1-score']:.4f}       {cls_data['support']}")
        
        print("-" * 70)
        print(f"{'Accuracy':<15}{report['accuracy']:.4f}")
        print(f"{'Macro Avg':<15}{report['macro avg']['precision']:.4f}       {report['macro avg']['recall']:.4f}       {report['macro avg']['f1-score']:.4f}       {report['macro avg']['support']}")
        print(f"{'Weighted Avg':<15}{report['weighted avg']['precision']:.4f}       {report['weighted avg']['recall']:.4f}       {report['weighted avg']['f1-score']:.4f}       {report['weighted avg']['support']}")
        
        # Save the report to a file
        with open(f'reports/{model_name_safe}_classification_report.txt', 'w') as f:
            f.write(f"{name} Classification Report\n")
            f.write("=" * (len(name) + 22) + "\n\n")
            f.write(f"{'Class':<15}{'Precision':<15}{'Recall':<15}{'F1-score':<15}{'Support':<10}\n")
            f.write("-" * 70 + "\n")
            
            for i, class_name in enumerate(present_class_names):
                if class_name in report:
                    cls_data = report[class_name]
                    f.write(f"{class_name:<15}{cls_data['precision']:.4f}       {cls_data['recall']:.4f}       {cls_data['f1-score']:.4f}       {cls_data['support']}\n")
            
            f.write("-" * 70 + "\n")
            f.write(f"{'Accuracy':<15}{report['accuracy']:.4f}\n")
            f.write(f"{'Macro Avg':<15}{report['macro avg']['precision']:.4f}       {report['macro avg']['recall']:.4f}       {report['macro avg']['f1-score']:.4f}       {report['macro avg']['support']}\n")
            f.write(f"{'Weighted Avg':<15}{report['weighted avg']['precision']:.4f}       {report['weighted avg']['recall']:.4f}       {report['weighted avg']['f1-score']:.4f}       {report['weighted avg']['support']}\n")
        
        # Plot confusion matrix with improved visibility
        plt.figure(figsize=(16, 14))  # Increased figure size
        cm = confusion_matrix(y_test, y_pred)
        
        # Enhanced heatmap
        heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=present_class_names, yticklabels=present_class_names,
                    annot_kws={"size": 14, "weight": "bold"})
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.title(f'Confusion Matrix - {name}', fontsize=22, fontweight='bold', pad=20)
        plt.xlabel('Predicted', fontsize=18, fontweight='bold', labelpad=15)
        plt.ylabel('Actual', fontsize=18, fontweight='bold', labelpad=15)
        
        plt.tight_layout()
        
        # Save the confusion matrix figure with high resolution
        plt.savefig(f'plots/confusion_matrix_{model_name_safe}.png', dpi=300, bbox_inches='tight')
    
    # Compare model performance with enhanced visualization
    plt.figure(figsize=(14, 8))  # Increased figure size
    bars = plt.bar(results.keys(), results.values())
    plt.title('Model Accuracy Comparison', fontsize=22, fontweight='bold', pad=20)
    plt.xlabel('Model', fontsize=18, fontweight='bold', labelpad=15)
    plt.ylabel('Accuracy', fontsize=18, fontweight='bold', labelpad=15)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the model comparison figure with high resolution
    plt.savefig('plots/model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    
    return results
        
# Main execution code - now as a sequence of steps instead of a function
if __name__ == "__main__":
    # Step 1: Generate credit rating data
    print("Step 1: Generating credit rating data...")
    df, class_names = generate_credit_rating_data(n_samples=10000)
    
    print("\nData overview:")
    print(df.head())
    print(f"\nData shape: {df.shape}")
    
    # Step 2: Prepare the data
    print("\nStep 2: Preparing the data...")
    X = df.drop(['credit_rating', 'credit_rating_label'], axis=1)
    y = df['credit_rating']
    feature_names = X.columns.tolist()
    
    # Important: Using stratify=y only if all classes have at least 2 samples
    class_counts = y.value_counts()
    min_count = class_counts.min()
    
    if min_count >= 2:
        print("Using stratified split since all classes have at least 2 samples")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        print("Using regular split since some classes have less than 2 samples")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 3: Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 4: Analyze class distribution
    print("\nStep 4: Analyzing class distribution...")
    class_counts = analyze_class_distribution(df, class_names)
    
    # Step 5: Create class mapping and reverse mapping
    # This is the missing part that needs to be added
    # We need to map the classes to contiguous integers for some ML algorithms and sampling methods
    unique_classes = sorted(np.unique(np.concatenate([y_train, y_test])))
    class_mapping = {original: i for i, original in enumerate(unique_classes)}
    reverse_mapping = {i: original for original, i in class_mapping.items()}
    
    # Apply the mapping to the training and test labels
    y_train_mapped = np.array([class_mapping[y] for y in y_train])
    y_test_mapped = np.array([class_mapping[y] for y in y_test])
    
    # Step 6: Analyze feature importance
    print("\nStep 5: Analyzing feature importance...")
    importances, indices = analyze_feature_importance(df, X_train_scaled, y_train_mapped, feature_names)
    
    # Step 7: Evaluate models before sampling
    print("\nStep 6: Evaluating models before sampling...")
    model_results, models = evaluate_models(X_train_scaled, y_train_mapped, X_test_scaled, y_test_mapped, class_names, reverse_mapping)
    
    # Step 8: Find the best model
    best_model_name = max(model_results, key=model_results.get)
    best_model = (best_model_name, models[best_model_name])
    print(f"\nBest model: {best_model_name} with accuracy {model_results[best_model_name]:.4f}")
    
    # Step 9: Apply sampling techniques
    print("\nStep 8: Applying sampling techniques...")
    resampled_data = apply_sampling_techniques(X_train_scaled, y_train_mapped, class_counts, reverse_mapping, class_names)
    
    # Step 10: Evaluate models after sampling
    print("\nStep 9: Evaluating models after sampling...")
    sampling_results = {}
    
    if resampled_data:
        sampling_results = evaluate_models_after_sampling(resampled_data, X_test_scaled, y_test_mapped, best_model, class_names, reverse_mapping)
        
        # Step 11: Analyze results
        print("\nStep 10: Analyzing results...")
        if sampling_results:
            best_sampling = max(sampling_results, key=sampling_results.get)
            print(f"Best sampling technique: {best_sampling} with accuracy {sampling_results[best_sampling]:.4f}")
        else:
            print("No sampling techniques produced valid results to evaluate.")
    else:
        print("No valid resampled data was produced. Skipping evaluation of sampling techniques.")
    
    # Compare best model before and after sampling
    print("\nComparison before and after sampling:")
    print(f"Best model before sampling: {best_model_name} with accuracy {model_results[best_model_name]:.4f}")
    
    improvement = 0
    if sampling_results:
        best_sampling = max(sampling_results, key=sampling_results.get)
        
        # Check if the accuracy after sampling is actually better
        if sampling_results[best_sampling] > model_results[best_model_name]:
            print(f"Best model after sampling: {best_model_name} with {best_sampling} with accuracy {sampling_results[best_sampling]:.4f}")
            # Improvement percentage
            improvement = (sampling_results[best_sampling] - model_results[best_model_name]) / model_results[best_model_name] * 100
            print(f"Improvement: {improvement:.2f}%")
        else:
            print(f"Best model after sampling: {best_model_name} with {best_sampling} with accuracy {sampling_results[best_sampling]:.4f}")
            print(f"Note: Sampling did not improve accuracy. Using the original model is recommended.")
            # Negative improvement (actually a decrease)
            improvement = (sampling_results[best_sampling] - model_results[best_model_name]) / model_results[best_model_name] * 100
            print(f"Accuracy change: {improvement:.2f}%")
    else:
        print("No valid sampling results to compare.")
    
    # Step 12: Summarize findings and recommendations
    print("\n=== Credit Rating Classification Findings ===")
    print(f"1. Best model: {best_model_name}")
    
    if sampling_results:
        best_sampling = max(sampling_results, key=sampling_results.get)
        print(f"2. Best sampling technique: {best_sampling}")
    else:
        best_sampling = "None (no successful sampling)"
        print(f"2. Best sampling technique: None (sampling techniques failed)")
    
    print(f"3. Features in order of importance:")
    for i in indices[:5]:
        print(f"   - {feature_names[i]}: {importances[i]:.4f}")
    print("4. Performance by credit rating class:")
    print("   (See classification reports for details)")
    print(f"5. Overall improvement with sampling: {improvement:.2f}%")
    
    # Create recommendations dictionary
    recommendations = {
        'best_model': best_model_name,
        'best_sampling': best_sampling,
        'important_features': [feature_names[i] for i in indices[:5]],
        'accuracy_improvement': improvement
    }
    
    # Save a summary of findings to a file
    with open('reports/credit_rating_classification_findings.txt', 'w') as f:
        f.write("=== Credit Rating Classification Findings ===\n")
        f.write(f"1. Best model: {best_model_name}\n")
        
        if sampling_results:
            best_sampling = max(sampling_results, key=sampling_results.get)
            f.write(f"2. Best sampling technique: {best_sampling}\n")
        else:
            f.write(f"2. Best sampling technique: None (sampling techniques failed)\n")
        
        f.write(f"3. Features in order of importance:\n")
        for i in indices[:5]:
            f.write(f"   - {feature_names[i]}: {importances[i]:.4f}\n")
        f.write("4. Performance by credit rating class:\n")
        f.write("   (See classification reports in the reports directory)\n")
        f.write(f"5. Overall improvement with sampling: {improvement:.2f}%\n")
    
    # Step 13: Print final recommendations
    print("\n=== Recommendations ===")
    print(f"1. Use {recommendations['best_model']} for credit rating classification")
    
    if improvement > 0:
        print(f"2. Apply {recommendations['best_sampling']} to address class imbalance")
    else:
        print(f"2. Use original data without sampling (sampling techniques did not improve accuracy)")
    
    print("3. Focus on these top features for data collection and feature engineering:")
    for i, feature in enumerate(recommendations['important_features']):
        print(f"   {i+1}. {feature}")
    
    if improvement > 0:
        print(f"4. Expected accuracy improvement: {recommendations['accuracy_improvement']:.2f}%")
    else:
        print(f"4. Note: Sampling resulted in {recommendations['accuracy_improvement']:.2f}% change in accuracy (negative)")
        print("   It is recommended to use the original model without sampling.")
    
    # Save recommendations to a file
    with open('reports/credit_rating_classification_recommendations.txt', 'w') as f:
        f.write("=== Recommendations ===\n")
        f.write(f"1. Use {recommendations['best_model']} for credit rating classification\n")
        
        if improvement > 0:
            f.write(f"2. Apply {recommendations['best_sampling']} to address class imbalance\n")
        else:
            f.write(f"2. Use original data without sampling (sampling techniques did not improve accuracy)\n")
        
        f.write("3. Focus on these top features for data collection and feature engineering:\n")
        for i, feature in enumerate(recommendations['important_features']):
            f.write(f"   {i+1}. {feature}\n")
        
        if improvement > 0:
            f.write(f"4. Expected accuracy improvement: {recommendations['accuracy_improvement']:.2f}%\n")
        else:
            f.write(f"4. Note: Sampling resulted in {recommendations['accuracy_improvement']:.2f}% change in accuracy (negative)\n")
            f.write("   It is recommended to use the original model without sampling.\n")
    
    print("\nAll analysis results, reports, and figures have been saved to disk."), models

# 5. Apply different sampling techniques
def apply_sampling_techniques(X_train, y_train, class_counts, reverse_mapping, class_names):
    """
    Apply different sampling techniques to address class imbalance.
    """
    # Count samples in smallest class
    unique_classes, class_sample_counts = np.unique(y_train, return_counts=True)
    min_samples = np.min(class_sample_counts)
    
    # Adjust neighbor parameters based on smallest class
    k_neighbors_param = max(1, min(3, min_samples - 1))  # At least 1, at most min_samples-1
    
    sampling_techniques = {
        'SMOTE': SMOTE(random_state=42, k_neighbors=k_neighbors_param),
        'ADASYN': ADASYN(random_state=42, n_neighbors=k_neighbors_param),
        'Random Under-sampling': RandomUnderSampler(random_state=42),
        'SMOTE-Tomek': SMOTETomek(random_state=42, smote=SMOTE(k_neighbors=k_neighbors_param))
    }
    
    print(f"Using k_neighbors={k_neighbors_param} for SMOTE-based techniques (based on min class size of {min_samples})")
    
    # Create a directory for sampling distribution plots
    import os
    if not os.path.exists('sampling_distributions'):
        os.makedirs('sampling_distributions')
    
    resampled_data = {}
    
    for name, sampler in sampling_techniques.items():
        print(f"\nApplying {name}...")
        try:
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            
            # Check if resampling produced a reasonable result
            if len(y_resampled) < 50:  # If we have too few samples after resampling
                print(f"Warning: {name} resulted in only {len(y_resampled)} samples, which may be too few for effective training.")
                # We'll still include it since it technically worked
            
            # Count the number of samples in each class after resampling
            unique, counts = np.unique(y_resampled, return_counts=True)
            resampled_counts = dict(zip(unique, counts))
            
            # Print resampling statistics
            print(f"Original shape: {X_train.shape}")
            print(f"Resampled shape: {X_resampled.shape}")
            
            # Calculate and print the change in class distribution
            print("\nClass distribution change:")
            for i in sorted(resampled_counts.keys()):
                # For the resampled data, we need to consider the mapping
                # The y_train has been mapped, so we need to map i back to original class
                original_class_idx = reverse_mapping.get(i, i)
                
                # Get the count from the original data using the original index
                original_count = np.sum(y_train == i)  # Count directly from y_train
                resampled_count = resampled_counts.get(i, 0)
                change = resampled_count - original_count
                
                # Get the class name if available
                class_label = class_names[original_class_idx] if original_class_idx < len(class_names) else f"Class {original_class_idx}"
                print(f"Class {i} ({class_label}): {original_count} -> {resampled_count} (Change: {change})")
            
            resampled_data[name] = (X_resampled, y_resampled)
            
            # Plot the class distribution after resampling with enhanced visualization
            plt.figure(figsize=(16, 8))  # Increased figure size
            class_labels = [class_names[reverse_mapping.get(i, i)] if reverse_mapping.get(i, i) < len(class_names) 
                            else f"Class {i}" for i in sorted(resampled_counts.keys())]
            
            bars = plt.bar(range(len(resampled_counts)), [resampled_counts.get(i, 0) for i in sorted(resampled_counts.keys())])
            plt.xticks(range(len(resampled_counts)), class_labels, rotation=45, ha='right')
            plt.title(f'Class Distribution After {name}', fontsize=22, fontweight='bold', pad=20)
            plt.xlabel('Credit Rating Class', fontsize=18, fontweight='bold', labelpad=15)
            plt.ylabel('Count', fontsize=18, fontweight='bold', labelpad=15)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 50,
                        f'{height}',
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            # Save the sampling distribution figure with high resolution
            plt.savefig(f'sampling_distributions/class_distribution_after_{name.replace(" ", "_").replace("-", "_")}.png', 
                        dpi=300, bbox_inches='tight')
            
        except Exception as e:
            print(f"Error applying {name}: {str(e)}")
            print(f"Skipping {name} sampling technique.")
    
    return resampled_data

# 6. Evaluate models after sampling
def evaluate_models_after_sampling(resampled_data, X_test, y_test, best_model, class_names, reverse_mapping):
    """
    Evaluate the best model after applying different sampling techniques.
    """
    # Since we've applied a class mapping, we need to adjust class_names accordingly
    present_class_names = []
    for i in range(len(np.unique(y_test))):
        original_class = reverse_mapping[i]
        if original_class < len(class_names):
            present_class_names.append(class_names[original_class])
        else:
            present_class_names.append(f"Class {original_class}")
    
    # Create a directory for sampling reports if it doesn't exist
    import os
    if not os.path.exists('sampling_reports'):
        os.makedirs('sampling_reports')
    
    results = {}
    
    # Use the best model from previous evaluation
    model_name = best_model[0]
    model_class = best_model[1].__class__
    
    for sampling_name, (X_resampled, y_resampled) in resampled_data.items():
        sampling_name_safe = sampling_name.replace(' ', '_').replace('-', '_')
        print(f"\nEvaluating {model_name} with {sampling_name}...")
        
        # Create a new instance of the model
        model = model_class()
        
        # Train the model on the resampled data
        model.fit(X_resampled, y_resampled)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[sampling_name] = accuracy
        
        # Print accuracy
        print(f"{sampling_name} accuracy: {accuracy:.4f}")
        
        # Format and save classification report
        report = classification_report(y_test, y_pred, target_names=present_class_names, zero_division=0, output_dict=True)
        
        # Print prettier classification report
        print("\nClassification Report:")
        print("-" * 70)
        print(f"{'Class':<15}{'Precision':<15}{'Recall':<15}{'F1-score':<15}{'Support':<10}")
        print("-" * 70)
        
        for i, class_name in enumerate(present_class_names):
            if class_name in report:
                cls_data = report[class_name]
                print(f"{class_name:<15}{cls_data['precision']:.4f}       {cls_data['recall']:.4f}       {cls_data['f1-score']:.4f}       {cls_data['support']}")
        
        print("-" * 70)
        print(f"{'Accuracy':<15}{report['accuracy']:.4f}")
        print(f"{'Macro Avg':<15}{report['macro avg']['precision']:.4f}       {report['macro avg']['recall']:.4f}       {report['macro avg']['f1-score']:.4f}       {report['macro avg']['support']}")
        print(f"{'Weighted Avg':<15}{report['weighted avg']['precision']:.4f}       {report['weighted avg']['recall']:.4f}       {report['weighted avg']['f1-score']:.4f}       {report['weighted avg']['support']}")
        
        # Save the report to a file
        with open(f'sampling_reports/{model_name.replace(" ", "_")}_{sampling_name_safe}_report.txt', 'w') as f:
            f.write(f"{model_name} with {sampling_name} Classification Report\n")
            f.write("=" * (len(model_name) + len(sampling_name) + 26) + "\n\n")
            f.write(f"{'Class':<15}{'Precision':<15}{'Recall':<15}{'F1-score':<15}{'Support':<10}\n")
            f.write("-" * 70 + "\n")
            
            for i, class_name in enumerate(present_class_names):
                if class_name in report:
                    cls_data = report[class_name]
                    f.write(f"{class_name:<15}{cls_data['precision']:.4f}       {cls_data['recall']:.4f}       {cls_data['f1-score']:.4f}       {cls_data['support']}\n")
            
            f.write("-" * 70 + "\n")
            f.write(f"{'Accuracy':<15}{report['accuracy']:.4f}\n")
            f.write(f"{'Macro Avg':<15}{report['macro avg']['precision']:.4f}       {report['macro avg']['recall']:.4f}       {report['macro avg']['f1-score']:.4f}       {report['macro avg']['support']}\n")
            f.write(f"{'Weighted Avg':<15}{report['weighted avg']['precision']:.4f}       {report['weighted avg']['recall']:.4f}       {report['weighted avg']['f1-score']:.4f}       {report['weighted avg']['support']}\n")
        
        # Plot confusion matrix with enhanced visualization
        plt.figure(figsize=(16, 14))  # Increased figure size
        cm = confusion_matrix(y_test, y_pred)
        
        # Enhanced heatmap for better readability
        heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=present_class_names, yticklabels=present_class_names,
                    annot_kws={"size": 14, "weight": "bold"})
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.title(f'Confusion Matrix - {model_name} with {sampling_name}', 
                  fontsize=22, fontweight='bold', pad=20)
        plt.xlabel('Predicted', fontsize=18, fontweight='bold', labelpad=15)
        plt.ylabel('Actual', fontsize=18, fontweight='bold', labelpad=15)
        
        plt.tight_layout()
        
        # Save the confusion matrix figure with high resolution
        plt.savefig(f'plots/confusion_matrix_{model_name.replace(" ", "_")}_{sampling_name_safe}.png', 
                    dpi=300, bbox_inches='tight')
    
    # Compare sampling techniques with enhanced visualization
    if results:
        plt.figure(figsize=(14, 8))  # Increased figure size
        bars = plt.bar(results.keys(), results.values())
        plt.title(f'Sampling Techniques Comparison with {model_name}', 
                  fontsize=22, fontweight='bold', pad=20)
        plt.xlabel('Sampling Technique', fontsize=18, fontweight='bold', labelpad=15)
        plt.ylabel('Accuracy', fontsize=18, fontweight='bold', labelpad=15)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     f'{height:.3f}',
                     ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the sampling comparison figure with high resolution
        plt.savefig('plots/sampling_techniques_comparison.png', dpi=300, bbox_inches='tight')
    
    return results

# Main execution code - now as a sequence of steps instead of a function
if __name__ == "__main__":
    # Step 1: Generate credit rating data
    print("Step 1: Generating credit rating data...")
    df, class_names = generate_credit_rating_data(n_samples=10000)
    
    print("\nData overview:")
    print(df.head())
    print(f"\nData shape: {df.shape}")
    
    # Step 2: Prepare the data
    print("\nStep 2: Preparing the data...")
    X = df.drop(['credit_rating', 'credit_rating_label'], axis=1)
    y = df['credit_rating']
    feature_names = X.columns.tolist()
    
    # Important: Using stratify=y only if all classes have at least 2 samples
    class_counts = y.value_counts()
    min_count = class_counts.min()
    
    if min_count >= 2:
        print("Using stratified split since all classes have at least 2 samples")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        print("Using regular split since some classes have less than 2 samples")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 3: Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 4: Analyze class distribution
    print("\nStep 4: Analyzing class distribution...")
    class_counts = analyze_class_distribution(df, class_names)
    
    # Step 5: Create class mapping and reverse mapping
    # This is the missing part that needs to be added
    # We need to map the classes to contiguous integers for some ML algorithms and sampling methods
    unique_classes = sorted(np.unique(np.concatenate([y_train, y_test])))
    class_mapping = {original: i for i, original in enumerate(unique_classes)}
    reverse_mapping = {i: original for original, i in class_mapping.items()}
    
    # Apply the mapping to the training and test labels
    y_train_mapped = np.array([class_mapping[y] for y in y_train])
    y_test_mapped = np.array([class_mapping[y] for y in y_test])
    
    # Step 6: Analyze feature importance
    print("\nStep 5: Analyzing feature importance...")
    importances, indices = analyze_feature_importance(df, X_train_scaled, y_train_mapped, feature_names)
    
    # Step 7: Evaluate models before sampling
    print("\nStep 6: Evaluating models before sampling...")
    model_results, models = evaluate_models(X_train_scaled, y_train_mapped, X_test_scaled, y_test_mapped, class_names, reverse_mapping)
    
    # Step 8: Find the best model
    best_model_name = max(model_results, key=model_results.get)
    best_model = (best_model_name, models[best_model_name])
    print(f"\nBest model: {best_model_name} with accuracy {model_results[best_model_name]:.4f}")
    
    # Step 9: Apply sampling techniques
    print("\nStep 8: Applying sampling techniques...")
    resampled_data = apply_sampling_techniques(X_train_scaled, y_train_mapped, class_counts, reverse_mapping, class_names)
    
    # Step 10: Evaluate models after sampling
    print("\nStep 9: Evaluating models after sampling...")
    sampling_results = {}
    
    if resampled_data:
        sampling_results = evaluate_models_after_sampling(resampled_data, X_test_scaled, y_test_mapped, best_model, class_names, reverse_mapping)
        
        # Step 11: Analyze results
        print("\nStep 10: Analyzing results...")
        if sampling_results:
            best_sampling = max(sampling_results, key=sampling_results.get)
            print(f"Best sampling technique: {best_sampling} with accuracy {sampling_results[best_sampling]:.4f}")
        else:
            print("No sampling techniques produced valid results to evaluate.")
    else:
        print("No valid resampled data was produced. Skipping evaluation of sampling techniques.")
    
    # Compare best model before and after sampling
    print("\nComparison before and after sampling:")
    print(f"Best model before sampling: {best_model_name} with accuracy {model_results[best_model_name]:.4f}")
    
    improvement = 0
    if sampling_results:
        best_sampling = max(sampling_results, key=sampling_results.get)
        
        # Check if the accuracy after sampling is actually better
        if sampling_results[best_sampling] > model_results[best_model_name]:
            print(f"Best model after sampling: {best_model_name} with {best_sampling} with accuracy {sampling_results[best_sampling]:.4f}")
            # Improvement percentage
            improvement = (sampling_results[best_sampling] - model_results[best_model_name]) / model_results[best_model_name] * 100
            print(f"Improvement: {improvement:.2f}%")
        else:
            print(f"Best model after sampling: {best_model_name} with {best_sampling} with accuracy {sampling_results[best_sampling]:.4f}")
            print(f"Note: Sampling did not improve accuracy. Using the original model is recommended.")
            # Negative improvement (actually a decrease)
            improvement = (sampling_results[best_sampling] - model_results[best_model_name]) / model_results[best_model_name] * 100
            print(f"Accuracy change: {improvement:.2f}%")
    else:
        print("No valid sampling results to compare.")
    
    # Step 12: Summarize findings and recommendations
    print("\n=== Credit Rating Classification Findings ===")
    print(f"1. Best model: {best_model_name}")
    
    if sampling_results:
        best_sampling = max(sampling_results, key=sampling_results.get)
        print(f"2. Best sampling technique: {best_sampling}")
    else:
        best_sampling = "None (no successful sampling)"
        print(f"2. Best sampling technique: None (sampling techniques failed)")
    
    print(f"3. Features in order of importance:")
    for i in indices[:5]:
        print(f"   - {feature_names[i]}: {importances[i]:.4f}")
    print("4. Performance by credit rating class:")
    print("   (See classification reports for details)")
    print(f"5. Overall improvement with sampling: {improvement:.2f}%")
    
    # Create recommendations dictionary
    recommendations = {
        'best_model': best_model_name,
        'best_sampling': best_sampling,
        'important_features': [feature_names[i] for i in indices[:5]],
        'accuracy_improvement': improvement
    }
    
    # Save a summary of findings to a file
    with open('reports/credit_rating_classification_findings.txt', 'w') as f:
        f.write("=== Credit Rating Classification Findings ===\n")
        f.write(f"1. Best model: {best_model_name}\n")
        
        if sampling_results:
            best_sampling = max(sampling_results, key=sampling_results.get)
            f.write(f"2. Best sampling technique: {best_sampling}\n")
        else:
            f.write(f"2. Best sampling technique: None (sampling techniques failed)\n")
        
        f.write(f"3. Features in order of importance:\n")
        for i in indices[:5]:
            f.write(f"   - {feature_names[i]}: {importances[i]:.4f}\n")
        f.write("4. Performance by credit rating class:\n")
        f.write("   (See classification reports in the reports directory)\n")
        f.write(f"5. Overall improvement with sampling: {improvement:.2f}%\n")
    
    # Step 13: Print final recommendations
    print("\n=== Recommendations ===")
    print(f"1. Use {recommendations['best_model']} for credit rating classification")
    
    if improvement > 0:
        print(f"2. Apply {recommendations['best_sampling']} to address class imbalance")
    else:
        print(f"2. Use original data without sampling (sampling techniques did not improve accuracy)")
    
    print("3. Focus on these top features for data collection and feature engineering:")
    for i, feature in enumerate(recommendations['important_features']):
        print(f"   {i+1}. {feature}")
    
    if improvement > 0:
        print(f"4. Expected accuracy improvement: {recommendations['accuracy_improvement']:.2f}%")
    else:
        print(f"4. Note: Sampling resulted in {recommendations['accuracy_improvement']:.2f}% change in accuracy (negative)")
        print("   It is recommended to use the original model without sampling.")
    
    # Save recommendations to a file
    with open('reports/credit_rating_classification_recommendations.txt', 'w') as f:
        f.write("=== Recommendations ===\n")
        f.write(f"1. Use {recommendations['best_model']} for credit rating classification\n")
        
        if improvement > 0:
            f.write(f"2. Apply {recommendations['best_sampling']} to address class imbalance\n")
        else:
            f.write(f"2. Use original data without sampling (sampling techniques did not improve accuracy)\n")
        
        f.write("3. Focus on these top features for data collection and feature engineering:\n")
        for i, feature in enumerate(recommendations['important_features']):
            f.write(f"   {i+1}. {feature}\n")
        
        if improvement > 0:
            f.write(f"4. Expected accuracy improvement: {recommendations['accuracy_improvement']:.2f}%\n")
        else:
            f.write(f"4. Note: Sampling resulted in {recommendations['accuracy_improvement']:.2f}% change in accuracy (negative)\n")
            f.write("   It is recommended to use the original model without sampling.\n")
    
    print("\nAll analysis results, reports, and figures have been saved to disk.")

    
    