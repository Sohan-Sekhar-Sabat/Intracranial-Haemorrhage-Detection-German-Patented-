from io import BytesIO
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from models import resnext101_32x8d_wsl
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models
from torchvision.transforms import CenterCrop,ToTensor,Compose

def get_cam(model, img):
    """
    Generates a Grad-CAM visualization for a given image using a pre-trained model.

    Args:
        model (torch.nn.Module): The pre-trained model used for generating Grad-CAM.
        img (torch.Tensor): The input image tensor.

    Returns:
        PIL.Image: The Grad-CAM visualization overlaid on the input image.
    """
    # Specify the target layers to visualize

    target_layers = [model.module.layer4[-1]]
    
    # Create the CAM extractor (SmoothGradCAM++ method)
    cam_extractor = SmoothGradCAMpp(model, target_layers)
    
    #Get Model output
    out = model(img)
    
    #Extract the activation map for the class with the highest score
    activ_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    
    #Overlay the CAM on the original image
    for name, cam in zip(cam_extractor.target_names, activ_map):
        vis = overlay_mask(to_pil_image(img.squeeze(0)), to_pil_image(cam, mode='F'), alpha=0.5)
    return vis

def get_model(path, n_classes):
    """
    Loads the pre-trained ResNeXt model and updates it for the number of target classes.

    Args:
        path (str): Path to the model checkpoint (.pth file).
        n_classes (int): The number of output classes for the model.

    Returns:
        torch.nn.Module: The loaded and updated model.
    """
    # Load the pre-trained ResNeXt model
    model = resnext101_32x8d_wsl()
    
    # Replace the fully connected layer to match the number of classes
    model.fc = torch.nn.Linear(2048, n_classes)
    
    # Use DataParallel to enable multi-GPU support
    model = nn.DataParallel(model)
    
     # Load the model weights from the checkpoint
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    
    
    return model

def main():
    """
    The main function to run the Streamlit application for ICH (Intracranial Hemorrhage) visualization.
    """
    # Set the page layout to Wide mode
    st.set_page_config(layout="wide")

    # Designing the interface
    st.title("ICH Visualiser")
    
    # For newline
    st.write("\n")
    
    # Set the columns for input image and for Grad-CAM visualization
    cols = st.columns((1, 1))
    cols[0].header("Input image")
    cols[-1].header("GRADCAM Visualisation")
    
    #idebar settings
    st.sidebar.title("Input selection")
    st.set_option("deprecation.showfileUploaderEncoding", False) #Disable file uploader encoding warning
    
    
    # Upload image file
    uploaded_file = st.sidebar.file_uploader("Upload files", type=["png", "jpeg", "jpg"])
    
    #If an image file is uploaded, display it in the first column
    if uploaded_file is not None:
        img = Image.open(BytesIO(uploaded_file.read()), mode="r").convert("RGB")
        #cols[0].image(img, use_column_width=True)
        cols[0].image(img, width=300)
        
    #Prediction button
    if st.sidebar.button("Predict"):
        #If no image is uploaded, show error message
        if uploaded_file is None:
            st.sidebar.error("Please upload an image first")

        else:
            with st.spinner("Analyzing..."):

                # Preprocess image(Center crop and convert to tensor)
                tfms_img = Compose([CenterCrop(200), ToTensor()])
                img_tensor = tfms_img(img).float()
                
                # List of possible labels for the output
                label_list = ["Epidural",
                            "Intraparenchymal",
                            "Intraventricular",
                            "Subarachnoid",
                            "Subdural",
                            "Any",
                        ]
                # Load the pre-trained model
                model=get_model('models/png_model_e10_final.pt',n_classes=6)
                
                
                # Forward pass the image through the model
                out = model(img_tensor.unsqueeze(0))
                out = torch.sigmoid(out) # Apply sigmoid to get probabilities
                out_trgt = torch.round(out) # Round the probabilities to get predictions
                
                # Convert the tensor to numpy for easier handling
                out_np = out_trgt.detach().numpy()
                print(out_np)
                preds = np.where(out_np == 1)[1] # Get the indices of positive predictions

                probas = torch.round((out) * 100).detach().numpy()[0]
                # print(probas)

                proba_ord = np.argsort(out.detach().numpy())[0][::-1]
                # print(proba_ord)

                label = []
                if len(out_np):
                    for i in preds:
                        label.append(label_list[i])
                else:
                    label.append("ICH Not Present")

                arg_s = {}
                for i in proba_ord:
                    arg_s[label_list[int(i)]] = probas[int(i)]
                       
                visualisation = get_cam(model,img_tensor.unsqueeze(0))
                #cols[-1].image(visualisation, use_column_width=True)
                cols[-1].image(visualisation, width=300)
                #df = pd.DataFrame(data=np.zeros((6,2)),
                     # columns=['Subtype','Predicted Probability'],
                     # index=np.linspace(1, 6, 6, dtype=int))
                df=pd.DataFrame(arg_s,index=[0])
                st.write('Confidence Levels(In Percentage)')
                st.table(data=df)
                if(len(label)==0):
                    final ='ICH Not Present'
                else:
                    final = label[0]
                #print()
                #new = pd.DataFrame(data=probas,columns=("epidural",
                #            "intraparenchymal",
                #            "intraventricular",
                #            "subarachnoid",
                #            "subdural",
                #            "any",
                #        ))
                #st.table(data=arg_s)
                st.write('Most Probable Conclusion:',final)

# Entry point for the script
if __name__ == "__main__":
    main()