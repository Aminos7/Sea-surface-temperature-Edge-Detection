# Sea Surface Temperature Edge Detection

This project is aimed at detecting the area that separates cold and warm areas in sea surface temperature images using state-of-the-art edge detection algorithms, including DexiNed, RCF, RINDnet, and HED.

## Goal
The main goal of this project is to apply edge detection algorithms to sea surface temperature images to identify the boundary between cold and warm areas accurately. The identified boundary can be useful in various applications, including climate research and oceanography.

## Training Dataset
To train the models, we generated sea surface temperature images and their masks from scratch using scripts available in the `src/scripts` folder. If you wish to use the pre-prepared dataset, it is available in the `Training_Dataset` folder.

## Usage
To run this project, you will need to follow these steps:
1. Clone this repository to your local machine.
2. Install the required packages by running `pip install -r requirements.txt`.
3. If you want to use your own dataset, generate your sea surface temperature images and their corresponding masks using the scripts provided in `src/scripts`. Otherwise, use the pre-prepared dataset in `Training_Dataset`.
4. Run the edge detection algorithms using the command `python edge_detection.py` with the desired algorithm as an argument (`DexiNed`, `RCF`, `RINDnet`, or `HED`).
5. The output will be saved to the `output` folder with the corresponding algorithm name as the prefix.

## Results
The edge detection algorithms have been evaluated on the generated dataset and have shown promising results in detecting the boundary between cold and warm areas in sea surface temperature images.

## Acknowledgements
We would like to acknowledge the following open-source projects that were used in this project:
- DexiNed: [https://github.com/xavysp/DexiNed](https://github.com/xavysp/DexiNed)
- RCF: [https://github.com/yun-liu/rcf](https://github.com/yun-liu/rcf)
- RINDnet: [https://github.com/xavysp/RINDnet](https://github.com/xavysp/RINDnet)
- HED: [https://github.com/s9xie/hed](https://github.com/s9xie/hed)
