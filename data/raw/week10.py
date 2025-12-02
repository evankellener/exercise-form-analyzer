# This comprehensive script serves as a complete template for the final capstone project.
# It includes functions for creating the project structure, building a full
# machine learning pipeline from data preprocessing to evaluation, and generating
# templates for the final project deliverables (presentation and reflection essay).

# Import all necessary libraries for the full pipeline.
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# --- Final Project Template Class ---
class FinalProjectTemplate:
    """
    A class to generate all the necessary components and templates for the final project.
    """
    def __init__(self, project_name, team_members, problem_description):
        # Initialize key project details.
        self.project_name = project_name
        self.team_members = team_members
        self.problem_description = problem_description
        self.model = None
        self.training_history = None
        self.project_metadata = {
            'created_date': datetime.now().isoformat(),
            'version': '1.0',
            'status': 'in_development'
        }
    
    def create_project_structure(self):
        """
        Creates a clean and organized folder structure for the project.
        This promotes good development practices.
        """
        folders = [
            'data/raw', 'data/processed', 'models', 'notebooks', 'src',
            'docs', 'results', 'presentation'
        ]
        for folder in folders:
            # os.makedirs() creates directories recursively. exist_ok=True prevents errors if they already exist.
            os.makedirs(folder, exist_ok=True)
            print(f"Created folder: {folder}")
        
        # Generates a professional README.md file template.
        readme_content = f"""
# {self.project_name}
...
"""
        with open('README.md', 'w') as f:
            f.write(readme_content)
        print("Project structure created successfully!")
    
    def create_complete_pipeline(self, input_shape, num_classes, class_names):
        """
        Generates the functions for a complete ML pipeline.
        This provides a roadmap for the technical part of the project.
        """
        # 1. Data preprocessing pipeline function.
        def preprocess_pipeline(image_path, target_size=(224, 224)):
            # Loads, resizes, and normalizes a single image.
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, target_size)
            # Apply basic data augmentation techniques (horizontal flip, rotation).
            augmented_images = []
            augmented_images.append(image / 255.0)
            flipped = cv2.flip(image, 1)
            augmented_images.append(flipped / 255.0)
            center = (image.shape[1]//2, image.shape[0]//2)
            rotation_matrix = cv2.getRotationMatrix2D(center, 15, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
            augmented_images.append(rotated / 255.0)
            return np.array(augmented_images)
        
        # 2. Model architecture function.
        def create_model():
            # Uses transfer learning with EfficientNetB0, a state-of-the-art model.
            base_model = tf.keras.applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
            base_model.trainable = False # Freeze the base model.
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
            return model
        
        # 3. Training configuration function.
        def train_model(model, X_train, y_train, X_val, y_val):
            # Configures the training process with callbacks for better performance and stability.
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy', 'top_3_accuracy']
            )
            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
                tf.keras.callbacks.ModelCheckpoint('models/best_model.h5', monitor='val_accuracy', save_best_only=True)
            ]
            history = model.fit(
                X_train, y_train, validation_data=(X_val, y_val), epochs=50,
                batch_size=32, callbacks=callbacks, verbose=1
            )
            return model, history
        
        # 4. Evaluation metrics function.
        def comprehensive_evaluation(model, X_test, y_test, class_names):
            # Provides a detailed evaluation report including classification report, confusion matrix, and visualizations.
            predictions = model.predict(X_test)
            predicted_classes = np.argmax(predictions, axis=1)
            report = classification_report(y_test, predicted_classes, target_names=class_names, output_dict=True)
            cm = confusion_matrix(y_test, predicted_classes)
            
            # Create a 2x2 subplot layout for the visualizations.
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot the confusion matrix using seaborn.
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax1)
            ax1.set_title('Confusion Matrix')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('Actual')
            
            # Plot the F1-score by class.
            class_accuracies = [report[class_name]['f1-score'] for class_name in class_names]
            ax2.bar(class_names, class_accuracies, color='skyblue')
            ax2.set_title('F1-Score by Class')
            ax2.set_ylabel('F1-Score')
            ax2.tick_params(axis='x', rotation=45)
            
            # Visualize sample predictions.
            sample_indices = np.random.choice(len(X_test), 6, replace=False)
            axes_flat = [plt.subplot(2, 3, i + 1) for i in range(6)]
            for i, idx in enumerate(sample_indices):
                ax = axes_flat[i]
                ax.imshow(X_test[idx])
                predicted_class = class_names[predicted_classes[idx]]
                true_class = class_names[y_test[idx]]
                confidence = predictions[idx][predicted_classes[idx]]
                color = 'green' if predicted_class == true_class else 'red'
                ax.set_title(f'Pred: {predicted_class}\nTrue: {true_class}\nConf: {confidence:.2f}', color=color, fontsize=8)
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig('results/evaluation_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return report, cm
        
        return preprocess_pipeline, create_model, train_model, comprehensive_evaluation
    
    def create_deployment_package(self):
        """
        Generates a stand-alone, production-ready prediction script and a requirements file.
        This simulates the deployment process.
        """
        # Generates a Python script with a predictor class.
        predictor_code = f'''
import tensorflow as tf
import cv2
import numpy as np
import json
import os

class {self.project_name.replace(" ", "")}Predictor:
    def __init__(self, model_path, metadata_path):
        self.model = tf.keras.models.load_model(model_path)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.class_names = self.metadata['class_names']
        self.input_shape = tuple(self.metadata['input_shape'])
    
    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_shape[:2])
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    
    def predict(self, image_path, return_probabilities=False):
        if not os.path.exists(image_path):
            return {{'error': f'Image file not found at {{image_path}}'}}
        processed_image = self.preprocess_image(image_path)
        predictions = self.model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        result = {{
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'class_index': int(predicted_class_idx)
        }}
        if return_probabilities:
            result['all_probabilities'] = {{
                class_name: float(prob) for class_name, prob in zip(self.class_names, predictions[0])
            }}
        return result
    
    def batch_predict(self, image_paths):
        results = []
        for image_path in image_paths:
            result = self.predict(image_path)
            result['image_path'] = image_path
            results.append(result)
        return results
'''
        with open('src/predictor.py', 'w') as f:
            f.write(predictor_code)
        
        # Generates a requirements.txt file with all necessary libraries.
        requirements = """
tensorflow>=2.8.0
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
Pillow>=8.3.0
"""
        with open('requirements.txt', 'w') as f:
            f.write(requirements)
        print("Deployment package created successfully!")
    
    def generate_presentation_outline(self):
        """Generates a detailed outline for the final video presentation."""
        presentation_outline = f"""
# Final Project Presentation: {self.project_name}
...
"""
        with open('presentation/presentation_outline.md', 'w') as f:
            f.write(presentation_outline)
        return presentation_outline
    
    def create_reflection_essay_template(self):
        """Generates a template for the final reflection essay."""
        essay_template = f"""
# Learning Journey Reflection: {self.project_name}
...
"""
        with open('docs/reflection_essay_template.md', 'w') as f:
            f.write(essay_template)
        return essay_template