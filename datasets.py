import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset

## A function to correct pixel data and rescale intercepts of 12 bit images
def dcm_correction(dcm_img):
    """
    Corrects pixel data for 12-bit grayscale DICOM images. This function adjusts pixel values
    to account for extra bits and sets a common RescaleIntercept value.

    Args:
        dcm_img (pydicom.dataset.FileDataset): DICOM image object with pixel data.

    Returns:
        None: Modifies the DICOM image object in place.
    """
    x = dcm_img.pixel_array + 1000
    px_mode = 4096
    x[x >= px_mode] = x[x >= px_mode] - px_mode #if there are extra bits in 12-bit grayscale(<=4096)
    dcm_img.PixelData = x.tobytes()
    dcm_img.RescaleIntercept = -1000 #setting a common value across all 12-bit US images
        
#Systemic/linear windowing
def window_image(dcm, window_center, window_width):
    """
    Applies windowing to DICOM images based on provided window center and width values.
    Optionally corrects the image if it is a 12-bit DICOM.

    Args:
        dcm (pydicom.dataset.FileDataset): DICOM image object with pixel data.
        window_center (int): The center of the window.
        window_width (int): The width of the window.

    Returns:
        np.ndarray: The windowed image as a NumPy array.
    """
    # Correct the image if it's a 12-bit grayscale DICOM
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        dcm_correction(dcm)
        
    # Reconstruct the image from pixel data
    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept #reconstructing the image from pixels
    img_min = window_center - window_width // 2 #lowest visible value
    img_max = window_center + window_width // 2 #highest visible value
    img = np.clip(img, img_min, img_max)

    return img

#Combining all
def bsb_window(dcm):
    """
    -------------------------Systemic Windowing--------------------------------------
    Combines multiple windowed DICOM images into a single image. The function applies
    three different windows (brain, subdural, and soft tissue) and merges them into a
    multi-channel image.

    Args:
        dcm (pydicom.dataset.FileDataset): DICOM image object with pixel data.

    Returns:
        np.ndarray: Combined image as a NumPy array with 3 channels (brain, subdural, soft tissue).
    """
    # Apply different windows to the DICOM image
    brain_img = window_image(dcm, 40, 80)
    subdural_img = window_image(dcm, 80, 200)
    soft_img = window_image(dcm, 40, 380)
    # Normalize the images
    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    soft_img = (soft_img - (-150)) / 380
    
    # Combine the images into a single multi-channel array
    bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1, 2, 0)

    return bsb_img


class IntracranialDataset(Dataset):
    """
    Custom PyTorch Dataset class for loading and processing intracranial images and labels
    from a CSV file. This class supports image augmentation and label handling.

    Args:
        csv_file (str): Path to the CSV file containing image IDs and labels.
        path (str): Directory path to the image files.
        labels (bool): Whether the dataset includes labels (True) or not (False).
        transform (callable, optional): Optional transformation to be applied on a sample.

    Attributes:
        path (str): Directory path to the image files.
        data (pd.DataFrame): DataFrame containing image IDs and labels.
        transform (callable): Transformation function for data augmentation.
        labels (bool): Whether the dataset includes labels or not.
    """
def __init__(self, csv_file, path, labels, transform=None):
        self.path = path
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = labels

def __len__(self):
    """
    Returns the length of the dataset.

    Returns:
        int: The number of samples in the dataset.
    """
    return len(self.data)

def __getitem__(self, idx):
    """
    Retrieves a sample from the dataset at the given index. The sample includes
        an image and, if labels are provided, corresponding labels.

    Args:
            idx (int): Index of the sample to retrieve.

    Returns:
            dict: A dictionary containing 'image_id', 'image', and (optionally) 'labels'.
    """
    # Retrieve the image ID from the CSV file
    img_id = self.data.loc[idx, 'Image']
    img_name = os.path.join(self.path, img_id + '.png')
    
    # Load the image from the specified path
    img = cv2.imread(img_name)   
#      img_id = self.data.loc[idx, 'ImageId']
     
     #try:
      #    img = pydicom.dcmread(self.path, img_id + '.dcm')
          #img = bsb_window(dicom)
      #except:
       #   img = np.zeros((512, 512, 3))
      
    if self.transform:       
          augmented = self.transform(image=img)
          img = augmented['image']   
    
    # If labels are present, return image along with labels      
    if self.labels:
        labels = torch.tensor(
              self.data.loc[idx, ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']])
        return {'image_id': img_id, 'image': img, 'labels': labels}         
    else:      
        # Otherwise, return only the image  
        return {'image_id': img_id, 'image': img}
