# --------------------------
# MAIN DATASET GENERATOR SCRIPT
# Generator script: From txt file to annotated augmented cipher generated images
# Usage: python generator.py
# No frequency analysis, all symbols are equally likely to be chosen

# Input: txt file
# Output: annotated augmented cipher generated images
# Output: JSON file with the transcription and the font characters
# Output: Transcribed text file
# Output: Dataset folder with the images
# Output: Dataset folder with the JSON files
# Output: Dataset folder with the transcribed text files
# --------------------------

from collections import defaultdict
import os
import json
import random
import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont

def transcription():
    """
    Creates dictionary to translate between regular text and the Copiale transcription
    
    Returns:
        Dictionary mapping regular text to transcription elements
    """
    # map of transcription with lists for multiple values
    transcribe = defaultdict(list)

    # Add mappings consistently
    transcribe['a'].extend(['p^.', 'n^.', 'h^.', 'Female'])
    transcribe['ä'].append('Female')
    transcribe['b'].append('SquareP')
    transcribe['c'].append('PhoenicianLetterPe')
    transcribe['d'].extend(['SmallPi', 'z'])
    transcribe['e'].extend(['a^^', 'e^^', 'i^^', 'o^^', 'u^^', 'LatinSmallLigatureFi', 'SleepingSymbol'])
    transcribe['f'].append('CapitalGamma')
    transcribe['g'].extend(['SmallDelta', 'x^.'])
    transcribe['h'].extend(['Saturn', 'RockSalt'])
    transcribe['i'].extend(['y^..', 'SmallNHook', 'SmallIota'])
    transcribe['j'].append('UpwardsArrow')
    transcribe['k'].append('RockSalt')
    transcribe['l'].append('c^.')
    transcribe['m'].append('+')
    transcribe['n'].extend(['m__', 'r__', 'n__', 'g'])
    transcribe['o'].extend(['Fire', 'o^.'])
    transcribe['ö'].append('SquaredRisingDiagonalSlash')
    transcribe['p'].append('d')
    transcribe['q'].append('qua')
    transcribe['r'].extend(['r^.', '3', 'j'])
    transcribe['s'].extend(['VerticalLine', 'SquaredPlus'])
    transcribe['t'].append('CapitalLambda')
    transcribe['u'].extend(['=', 'NotEqualTo'])
    transcribe['ü'].append('LatinLongLigatureFi')
    transcribe['v'].append('Earth')
    transcribe['w'].append('m^.')
    transcribe['x'].append('f')
    transcribe['y'].append('Infinity')
    transcribe['z'].append('s^.')

    # Syllables
    transcribe['sch'].append('Dagger')
    transcribe['ss'].append('SquaredPlus')
    transcribe['st'].append('TopHalfIntegral')
    transcribe['ch'].append('NorthEastArrow')
    transcribe['en'].append('u__')
    transcribe['em'].append('u__')

    # Repeat previous consonant ':' Applied later

    # Spaces
    transcribe[' '].extend([
        'Integral', 'a', 'b', 'c', 'd', 'e', 'f', 'TF', 'ScriptSmallG', 'h', 'i',
        'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'Integral', 't', 'u',
        'v', 'w', 'x', 'y', 'ScriptSmallZ', 'BigA', 'BigB', 'BigC', 'BigD', 
        'BigE', 'BigF', 'BigG', 'BigH', 'BigI', 'BigJ', 'BigK', 'BigL', 'BigM', 
        'BigN', 'BigO', 'BigP', 'BigQ', 'BigR', 'BigS', 'BigT', 'BigU', 'BigV', 
        'BigW', 'BigX', 'BigY', 'BigZ'
    ])

    return transcribe

def random_choice(symbols):
    """
    Select a transcription symbol randomly from available options
    
    Args:
        symbols (list): List of possible symbols to choose from
        
    Returns:
        str: Selected symbol
    """
    if not symbols:
        return ""
    return random.choice(symbols)

def apply_transcription(text, trans_dict):
    """
    Apply transcription to plain text using random selection when multiple options exist
    
    Args:
        text (str): Plain text to convert
        trans_dict (dict): Transcription dictionary
        
    Returns:
        str: Transcribed text
    """
    text = text.lower()
    result = []
    i = 0
    vowels = 'aeiouäöüy'  # including y and german umlauts as vowels
    
    syllables = ['sch', 'ss', 'st', 'ch', 'en', 'em']

    # First pass: identify direct consonant repetitions in the original text
    repeated_consonant_positions = set()
    j = 0
    while j < len(text) - 1:  # Stop one character before the end
        if (text[j].isalpha() and text[j] not in vowels and
            text[j] == text[j+1]):  # Current char equals next char
            repeated_consonant_positions.add(j+1)  # Mark the second position
        j += 1
    
    # Second pass: do the actual transcription
    while i < len(text):
        # Check for syllables first
        matched = False
        for syl in syllables:
            if text[i:i+len(syl)] == syl and i+len(syl) <= len(text):
                # Use random choice for syllables
                if syl in trans_dict and trans_dict[syl]:
                    result.append(random_choice(trans_dict[syl]))
                    matched = True
                    i += len(syl)
                    break
        
        if matched:
            continue

        # Handle individual characters
        char = text[i]
        
        # If this position was marked as a repeated consonant, use ':'
        if i in repeated_consonant_positions:
            result.append(':')
        elif char in trans_dict and trans_dict[char]:
            # Use random choice for characters
            result.append(random_choice(trans_dict[char]))
        else:
            result.append(char)

        i += 1

    return ' '.join(result)


def createTranslator():
    """
    Creates dictionary to translate between the transcription and the font (CopialeV2.ttf)
    
    Returns:
        Dictionary mapping transcription elements to font characters
    """
    # Dictionary with default values
    translator = defaultdict(lambda: '')
    
    # Letters
    translator['a'] = 'a'
    translator['a^^'] = 'A'
    translator['BigA'] = 'a'  # No direct translation
    translator['b'] = 'b'
    translator['BigB'] = 'b'  # No direct translation
    translator['c'] = 'c'
    translator['c^.'] = 'C'
    translator['BigC'] = 'c'  # No direct translation
    translator['d'] = 'd'
    translator['BigD'] = 'd'  # No direct translation
    translator['e'] = 'e'
    translator['e^^'] = 'E'
    translator['BigE'] = 'e'  # No direct translation
    translator['f'] = 'f'
    translator['BigF'] = '_'  # More or less
    translator['g'] = 'g'
    translator['BigG'] = 'g'  # No direct translation
    translator['h'] = 'h'
    translator['h^.'] = 'H'
    translator['BigH'] = 'h'  # No direct translation
    translator['i'] = 'i'
    translator['i^^'] = 'I'
    translator['BigI'] = 'i'  # No direct translation
    translator['j'] = 'j'
    translator['BigJ'] = 'j'  # No direct translation
    translator['k'] = 'k'
    translator['BigK'] = 'k'  # No direct translation
    translator['l'] = 'l'
    translator['BigL'] = 'l'  # No direct translation
    translator['m'] = 'm'
    translator['m^.'] = 'M'
    translator['m__'] = 'B'
    translator['BigM'] = 'm'  # No direct translation
    translator['n'] = 'n'
    translator['n^.'] = 'N'
    translator['n__'] = 'D'
    translator['BigN'] = 'n'  # No direct translation
    translator['o'] = 'o'
    translator['o^.'] = '&'
    translator['o^^'] = 'O'
    translator['BigO'] = 'o'  # No direct translation
    translator['p'] = 'p'
    translator['p^.'] = 'P'
    translator['BigP'] = 'p'  # No direct translation
    translator['q'] = 'q'
    translator['BigQ'] = 'q'  # No direct translation
    translator['r'] = 'r'
    translator['r^.'] = 'R'
    translator['r__'] = 'F'
    translator['BigR'] = 'r'  # No direct translation
    translator['s'] = 's'
    translator['s^.'] = 'S'
    translator['BigS'] = 's'  # No direct translation
    translator['t'] = 't'
    translator['BigT'] = 't'  # No direct translation
    translator['u'] = 'u'
    translator['u^^'] = 'U'
    translator['u__'] = 'G'
    translator['BigU'] = 'u'  # No direct translation
    translator['v'] = 'v'
    translator['BigV'] = 'v'  # No direct translation
    translator['w'] = 'w'
    translator['BigW'] = 'w'  # No direct translation
    translator['x'] = 'x'
    translator['x^.'] = 'X'
    translator['BigX'] = 'x'  # No direct translation
    translator['y'] = 'y'  
    translator['y^..'] = 'y'
    translator['BigY'] = 'y'  # No direct translation
    translator['z'] = 'z'
    translator['BigZ'] = 'z'  # No direct translation
    
    # Symbols
    translator['+'] = '+'
    translator['.'] = '.'
    translator['..'] = '..'
    translator['...'] = ','
    translator[':'] = ':'
    translator['='] = '='
    translator['3'] = '3'
    
    # Logograms
    translator['Alkali'] = '9'
    translator['BallotScriptX'] = '%'
    translator['BigFire'] = '<'  # No direct translation
    translator['BigInsularD'] = 'L'  # No direct translation
    translator['CapitalGamma'] = '~'
    translator['CapitalLambda'] = '^'
    translator['CircledEquals'] = '@'
    translator['Dagger'] = 'T'
    translator['Earth'] = '1'
    translator['Eye'] = '2'
    translator['Female'] = '0'
    translator['Fire'] = '<'
    translator['Infinity'] = '8'
    translator['InsularD'] = 'L'
    translator['Integral'] = '`'
    translator['LatinLongLigatureFi'] = ']'
    translator['LatinSmallLigatureFi'] = ')'
    translator['NorthEastArrow'] = '/'
    translator['NotEqualTo'] = '"'
    translator['PhoenicianLetterPe'] = '?'
    translator['RockSalt'] = '5'
    translator['Saturn'] = '-'
    translator['ScriptSmallG'] = 'K'
    translator['ScriptSmallZ'] = 'J'
    translator['SleepingSymbol'] = 'Z'
    translator['SmallDelta'] = '6'
    translator['SmallIota'] = '!'
    translator['SmallNHook'] = 'Y'
    translator['SmallPi'] = '>'
    translator['SquareP'] = 'Q'
    translator['SquaredPlus'] = '['
    translator['SquaredRisingDiagonalSlash'] = 'W'
    translator['TopHalfIntegral'] = '7'
    translator['TriangleDot'] = '#'
    translator['UpwardsArrow'] = '4'
    translator['VerticalLine'] = '|'
    
    # Copiale V2
    translator['(:'] = '('
    translator[':)'] = '$'
    translator['gate'] = '\''
    translator['Cloud'] = ';'
    translator['Pentagram'] = '*'
    
    # Not found in trainset, in font
    translator['TF'] = '\\'
    
    # Corrections
    translator['e?'] = '3'  # Assumed wrong direction
    translator['qua'] = 'W'  # Assumed missing line
    translator['('] = '('  # Assumed missing points
    translator[')'] = '$'  # Assumed missing points
    
    # Space
    translator[' '] = ' '
    
    # Add any missing punctuation
    translator['!'] = '!'
    translator['?'] = '?'
    translator[','] = ','
    
    return translator

def text_to_copiale(input_text):
    """
    Convert regular text to Copiale cipher
    
    Args:
        input_text (str): The text to convert to Copiale
        
    Returns:
        str: The transcribed text in Copiale notation
    """
    # Get the transcription dictionary
    trans_dict = transcription()
    
    # Initialize result
    result = []
    
    # Process each character
    i = 0
    while i < len(input_text):
        # Check for multi-character sequences first
        found = False
        for key_length in range(6, 1, -1):  # Check from 6-char to 2-char sequences
            if i + key_length <= len(input_text):
                potential_key = input_text[i:i+key_length].lower()
                if potential_key in trans_dict:
                    result.append(random_choice(trans_dict[potential_key]))
                    i += key_length
                    found = True
                    break
        
        # If no multi-character sequence found, process single character
        if not found:
            char = input_text[i]
            
            # Handle uppercase letters
            if char.isalpha() and char.isupper():
                # First check if the uppercase letter is directly in the dictionary
                if char in trans_dict:
                    result.append(random_choice(trans_dict[char]))
                else:
                    # Try the "Big" + uppercase version
                    big_key = char.upper()  # Make sure it's uppercase
                    result.append(random_choice(trans_dict[big_key]))
            # Handle lowercase and other characters
            elif char.lower() in trans_dict:
                result.append(random_choice(trans_dict[char.lower()]))
            else:
                # Keep characters that don't have a mapping
                result.append(char)
            
            i += 1
    
    return ' '.join(result)

def copiale_to_font(copiale_text):
    """
    Convert Copiale transcription to CopialeV2 font characters
    
    Args:
        copiale_text (str): The Copiale transcription text
        
    Returns:
        str: Text with characters mapped to CopialeV2 font
    """
    # Get the translator dictionary
    translator = createTranslator()
    
    # Split by spaces and translate each element
    transcription_elements = copiale_text.split()
    font_chars = []
    
    for element in transcription_elements:
        if element in translator:
            font_chars.append(translator[element])
        else:
            # Keep original if no translation exists
            font_chars.append(element)
    
    return ''.join(font_chars)  # Join without spaces for the font


def complete_translation(input_text):
    """
    Complete pipeline that:
    1. Converts input text to Copiale transcription
    2. Converts transcription to CopialeV2 font characters
    
    Args:
        input_text (str): The input text to convert
        
    Returns:
        tuple: (copiale_transcription, font_characters)
    """
    copiale_transcription = text_to_copiale(input_text)
    font_characters = copiale_to_font(copiale_transcription)
    
    return copiale_transcription, font_characters


def augmentor(img):
    TH,TW=img.shape

    param_gamma_low=.3
    #param_gamma_low=.5 # Nacho fixed
    param_gamma_high=2

    param_mean_gaussian_noise=0
    param_sigma_gaussian_noise=100**0.5

    param_kanungo_alpha=2 # params controlling how much foreground and background pixels flip state
    param_kanungo_beta=2
    param_kanungo_alpha0=1
    param_kanungo_beta0=1
    param_kanungo_mu=0
    param_kanungo_k=2

    param_min_shear=-.5 # here a little bit more shear to the left than to the right
    param_max_shear=.25

    param_rotation=3 # plus minus angles for rotation

    param_scale=.2 # one plus minus parameter as scaling factor

    param_movement_BB=6 # translation for cropping errors in pixels

    # add gaussian noise
    gauss = np.random.normal(param_mean_gaussian_noise,param_sigma_gaussian_noise,(TH,TW))
    gauss = gauss.reshape(TH,TW)
    gaussiannoise = np.uint8(np.clip(np.float32(img) + gauss,0,255))

    # randomly erode, dilate or nothing
    # we could move it also after binarization
    kernel=np.ones((3,3),np.uint8)
    #a=random.choice([1,2,3])
    a=random.choice([2,3]) # Nacho fixed
    #a = 3 # Nacho fixed
    if a==1:
        gaussiannoise=cv2.dilate(gaussiannoise,kernel,iterations=1)
    elif a==2:
        gaussiannoise=cv2.erode(gaussiannoise,kernel,iterations=1)

    # add random gamma correction
    gamma=np.random.uniform(param_gamma_low,param_gamma_high)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    gammacorrected = cv2.LUT(np.uint8(gaussiannoise), table)

    # binarize image with Otsu
    otsu_th,binarized = cv2.threshold(gammacorrected,0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Kanungo noise
    dist = cv2.distanceTransform(1-binarized, cv2.DIST_L1, 3)  # try cv2.DIST_L1 for newer versions of OpenCV
    dist2 = cv2.distanceTransform(binarized, cv2.DIST_L1, 3) # try cv2.DIST_L1 for newer versions of OpenCV

    dist = dist.astype('float64') # Tro add
    dist2 = dist2.astype('float64') # Tro add

    P=(param_kanungo_alpha0*np.exp(-param_kanungo_alpha * dist**2)) + param_kanungo_mu
    P2=(param_kanungo_beta0*np.exp(-param_kanungo_beta * dist2**2)) + param_kanungo_mu
    distorted=binarized.copy()
    distorted[((P>np.random.rand(P.shape[0],P.shape[1])) & (binarized==0))]=1
    distorted[((P2>np.random.rand(P.shape[0],P.shape[1])) & (binarized==1))]=0
    closing = cv2.morphologyEx(distorted, cv2.MORPH_CLOSE, np.ones((param_kanungo_k,param_kanungo_k),dtype=np.uint8))

    # apply binary image as mask and put it on a larger canvas
    pseudo_binarized = closing * (255-gammacorrected)
    canvas=np.zeros((3*TH,3*TW),dtype=np.uint8)
    canvas[TH:2*TH,TW:2*TW]=pseudo_binarized
    points=[]
    count = 0 # Tro add
    while(len(points)<1):
        count += 1 # Tro add
        if count > 50: # Tro add
            break # Tro add

        # random shear
        shear_angle=np.random.uniform(param_min_shear,param_max_shear)
        M=np.float32([[1,shear_angle,0],[0,1,0]])
        sheared = cv2.warpAffine(canvas,M,(3*TW,3*TH),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_CUBIC)

        # random rotation
        M = cv2.getRotationMatrix2D((3*TW/2,3*TH/2),np.random.uniform(-param_rotation,param_rotation),1)
        rotated = cv2.warpAffine(sheared,M,(3*TW,3*TH),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_CUBIC)

        # random scaling
        scaling_factor=np.random.uniform(1-param_scale,1+param_scale)
        scaled = cv2.resize(rotated,None,fx=scaling_factor,fy=scaling_factor,interpolation=cv2.INTER_CUBIC)

        # detect cropping parameters
        points = np.argwhere(scaled!=0)
        points = np.fliplr(points)

    if len(points) < 1: # Tro add
        return pseudo_binarized

    r = cv2.boundingRect(np.array([points]))

    #random cropping
    deltax=random.randint(-param_movement_BB,param_movement_BB)
    deltay=random.randint(-param_movement_BB,param_movement_BB)
    x1=min(scaled.shape[0]-1,max(0,r[1]+deltax))
    y1=min(scaled.shape[1]-1,max(0,r[0]+deltay))
    x2=min(scaled.shape[0],x1+r[3])
    y2=min(scaled.shape[1],y1+r[2])
    final_image=np.uint8(scaled[x1:x2,y1:y2])

    return final_image


def create_copiale_images_from_txt(txt_path, font_path, output_dir):
    """
    Generates images from each line in a .txt file containing Copiale transcriptions.
    
    Args:
        txt_path (str): Path to the input .txt file
        font_path (str): Path to the Copiale font file
        output_dir (str): Directory to save the images

    Returns:
        dict: Mapping of image filename to { transcription, copiale_font }
    """
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(txt_path))[0]
    dataset = {}

    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        font = ImageFont.truetype(font_path, 36)

        for i, line in enumerate(lines, 1):
            image_filename = f"{base_name}.png"
            image_path = os.path.join(output_dir, image_filename)

            copiale_text = copiale_to_font(line)
            create_text_image(copiale_text, font, image_path)


            dataset[image_filename] = {
                "transcription": line,
                "copiale_font": copiale_to_font(line)
            }

        return dataset

    except Exception as e:
        print(f"Error processing file '{txt_path}': {e}")
        return {}

def create_text_image(text, font, output_path, apply_augmentation=True):
    # Calculate text size
    dummy_img = Image.new('RGB', (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    left, top, right, bottom = dummy_draw.textbbox((0, 0), text, font=font)
    text_width = right - left
    text_height = bottom - top

    padding = 20
    width = text_width + (padding * 2)
    height = text_height + (padding * 2)

    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    draw.text((padding, padding), text, font=font, fill='black')

    # Save initial image
    image.save(output_path)

    if apply_augmentation:
        # Load image as grayscale for augmentation
        img = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
        augmented = augmentor(img)
        # Apply negative transformation
        augmented = cv2.bitwise_not(augmented)
        # Save the augmented image (overwriting the original or add suffix)
        cv2.imwrite(output_path, augmented)

def process_plaintext_file(input_path, font_path, output_dir, output_json_path, apply_augmentation=True):
    """
    Process a plain text file, generating Copiale transcription, font characters, and images
    
    Args:
        input_path (str): Path to the input plain text file
        font_path (str): Path to the Copiale font file
        output_dir (str): Directory to save the images
        output_json_path (str): Path to save the output JSON file
        apply_augmentation (bool): Whether to apply augmentation to images
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    # Initialize dictionaries and load font
    trans_dict = transcription()
    
    # Load font for rendering
    try:
        font = ImageFont.truetype(font_path, 36)
    except Exception as e:
        print(f"Error loading font '{font_path}': {e}")
        return
    
    # Read input file
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading input file '{input_path}': {e}")
        return
    
    # Get base name for file naming
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    dataset = {}
    
    # Process each line
    for i, plain_text in enumerate(lines, 1):
        # Generate unique filename for this line's image
        image_filename = f"{base_name}_{i:03d}.png"
        image_path = os.path.join(output_dir, image_filename)
        
        # Generate transcription and font characters
        copiale_transcription = apply_transcription(plain_text, trans_dict)
        font_characters = copiale_to_font(copiale_transcription)
        
        # Create and save the image
        create_text_image(font_characters, font, image_path, apply_augmentation)
        
        # Add to dataset
        dataset[image_filename] = {
            "plaintext": plain_text,
            "transcription": copiale_transcription,
            "copiale_font": font_characters
        }
    
    # Save dataset to JSON
    with open(output_json_path, 'w', encoding='utf-8') as out_file:
        json.dump(dataset, out_file, indent=2, ensure_ascii=False)
    
    print(f"Processed {len(dataset)} lines from '{input_path}'")
    print(f"Generated {len(dataset)} images in '{output_dir}'")
    print(f"Saved dataset to '{output_json_path}'")


def create_transcribed_txt(input_path, output_path):
    """
    Creates a new text file with the Copiale transcriptions of the input text
    
    Args:
        input_path (str): Path to the input plain text file
        output_path (str): Path where to save the transcribed text file
    """
    # Get the transcription dictionary
    trans_dict = transcription()
    
    try:
        # Read input file
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Process each line and write to output
        with open(output_path, 'w', encoding='utf-8') as out_file:
            for line in lines:
                # Apply transcription
                transcribed = apply_transcription(line, trans_dict)
                # Write to output file
                out_file.write(transcribed + '\n')
                
        print(f"Successfully created transcribed file at: {output_path}")
        return True
    
    except Exception as e:
        print(f"Error processing files: {e}")
        return False
    

def main():
    # Hardcoded paths 
    input_file = ""
    font_path = ""
    output_dir = ""
    output_json = ""
    apply_augmentation = True  # Set to False to disable augmentation

    process_plaintext_file(
        input_file,
        font_path,
        output_dir,
        output_json,
        apply_augmentation
    )

if __name__ == "__main__":
    main()
