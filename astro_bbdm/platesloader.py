import numpy as np
from astropy.io import fits
from astropy import units as u
from tqdm import tqdm 
from astroquery.skyview import SkyView 
from astropy.coordinates import Longitude, Latitude, Angle
from astroquery.hips2fits import hips2fits
from io import BytesIO
import warnings
import random


def check_plate(coords, survey1='PanSTARRS/DR1/r', survey2='CDS/P/SDSS9/r', fov_deg=.25, pixel_width=64):
    """
    Check whether two survey contain plate at a given coords

    Parameters:
    - survey1 (str): The name of the first survey.
    - survey2 (str): The name of the second survey
    - fov_deg (float): The Field of View in degrees.
    - pixel_width (int): The desired width (and height) in pixels.

    Returns:
    - 'ok' if surveys contain a plate.
    - 'not ok' if not.
    """
    # verbose=False silences the output during the check
    ps_plate = download_fits_as_numpy(
                survey_name=survey1,
                ra_deg=coords[0],
                dec_deg=coords[1],
                fov_deg=fov_deg,
                pixel_width=pixel_width,
                verbose=False
            )
    sdss_plate = download_fits_as_numpy(
                survey_name=survey2,
                ra_deg=coords[0],
                dec_deg=coords[1],
                fov_deg=fov_deg,
                pixel_width=pixel_width,
                verbose=False
            )
    
    # Changed .all() to .any() to ensure valid images with some black pixels are not rejected
    if (ps_plate.any()) and (sdss_plate.any()):
        return 'ok'
    else:
        return 'not ok'

        
def get_plates_coords(n_plates=10, survey1='PanSTARRS/DR1/r', survey2='CDS/P/SDSS9/r', fov_deg=.25, pixel_width=64):
    """
    Collects a list of coords of plates that are present in both survey1 and survey2
    
    Parameters:
    - n_plates (int): Number of plates to collect.
    - survey1 (str): The name of the first survey.
    - survey2 (str): The name of the second survey.
    - fov_deg (float): The Field of View in degrees.
    - pixel_width (int): The desired width (and height) in pixels.

    Returns:
    - list of collected coordinates.
    """
    plates = []
    collected = 0
    
    # Initialize tqdm progress bar
    # total=n_plates corresponds to the number of successful finds required
    with tqdm(total=n_plates, desc="Collecting Plates", unit="plate") as pbar:
        while collected < n_plates:
            ra = get_float(0, 360, 4)
            dec = get_float(-90, 90, 4)
            coords = (ra, dec) 

            ok = check_plate(coords, survey1=survey1, survey2=survey2, fov_deg=fov_deg, pixel_width=pixel_width)
            
            if ok == 'ok':
                plates.append(coords)
                collected += 1
                pbar.update(1) # Update bar only when a valid plate is found
            else:
                pass
                
    return plates


def download_fits_as_numpy(survey_name, ra_deg, dec_deg, fov_deg, pixel_width, verbose=True): # <--- FIXED SIGNATURE
    """
    Downloads a FITS image cutout from the HiPS2FITS service as a NumPy array.
    This version robustly checks all FITS extensions for the image data.

    Parameters:
    - survey_name (str): The name of the HiPS survey (e.g., 'CDS/P/HLA/SDSSr').
    - ra_deg (float): Right Ascension of the cutout center in degrees.
    - dec_deg (float): Declination of the cutout center in degrees.
    - fov_deg (float): The Field of View (size of the largest dimension) in degrees.
    - pixel_width (int): The desired width (and height) of the output image in pixels.
    - verbose (bool): If True, prints status messages.
    
    Returns:
    - numpy.ndarray or None: The image data as a 2D NumPy array, or zeros if failed.
    """
    
    # Define parameters with astropy units and objects
    ra_coord = Longitude(ra_deg * u.deg)
    dec_coord = Latitude(dec_deg * u.deg)
    fov_angle = Angle(fov_deg * u.deg)

    if verbose:
        print(f"Querying HiPS survey: {survey_name} at RA={ra_deg:.4f}, Dec={dec_deg:.4f}...")
    
    try:
        # Suppress warnings that may arise from using PYTHONHTTPSVERIFY=0 (SSL bypass)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            
            # 1. Query the hips2fits service. It returns the parsed astropy.io.fits.HDUList object.
            hdul = hips2fits.query(
                hips=survey_name,
                width=pixel_width,
                height=pixel_width,
                ra=ra_coord,
                dec=dec_coord,
                fov=fov_angle,
                projection="TAN",
                format='fits',
                get_query_payload=False
            )

        # 2. Process the HDUList object to find the image data
        if isinstance(hdul, fits.HDUList):
            image_data = None
            
            # --- Robust Data Extraction: Iterate through HDUs ---
            for i, hdu in enumerate(hdul):
                # Check if the HDU contains actual pixel data (NAXIS must be >= 2 for an image)
                # and that the data array is not None
                if hdu.data is not None and hdu.header.get('NAXIS', 0) >= 2:
                    image_data = hdu.data
                    if verbose:
                        print(f"Image data successfully found in HDU index: {i}")
                    break
            # ---------------------------------------------------
            
            hdul.close() # Always close the HDUList when done

            if image_data is not None and isinstance(image_data, np.ndarray):
                # Check if the data is all NaNs (still possible if coverage is sparse)
                if np.isnan(image_data).all():
                    if verbose:
                        print("Warning: Image array found, but all pixel values are NaN. This indicates no valid data coverage at these coordinates/FOV.")
                    return np.zeros((pixel_width, pixel_width))
                    
                if verbose:
                    print(f"Download successful. Data shape: {image_data.shape}")
                return image_data
            else:
                if verbose:
                    print("Error: Could not find a valid 2D image data array in any FITS extension.")
                return np.zeros((pixel_width, pixel_width))
        else:
            if verbose:
                print(f"Error: Expected an HDUList object, but received {type(hdul)}.")
            return np.zeros((pixel_width, pixel_width))

    except Exception as e:
        if verbose:
            print(f"An error occurred during download: {e}")
        return np.zeros((pixel_width, pixel_width))
        

def get_float(start, end, precision):
    return round(random.uniform(start, end), precision)