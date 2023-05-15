# Cross Stitch Color Calculator

This repository contains a Python app that calculates the most common colors in an input image using the KMeans algorithm. The app is designed to assist in embroidery pattern designing by associating RGB color values with commercially available cross stitch threads from DMC and Anchor companies.

## Usage

1. Clone the repository: `git clone https://github.com/senajoaop/crossstitch-color.git`
2. Navigate to the project directory: `cd crossstitch-color`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the Python script: `python cross_stitch_color.py`
5. Provide the path to the input image file when prompted.
6. The script will process the image using the KMeans algorithm to identify the most common colors.
7. The app will then associate these colors with cross stitch threads from DMC and Anchor companies.

Please note that the app requires the following Python modules: Scikit Learn and OpenCV. These dependencies are listed in the `requirements.txt` file and will be installed during the setup process.

## Contributing

Contributions to this repository are welcome! If you have any improvements, suggestions, or bug fixes, please feel free to contribute by following these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add your commit message"`
4. Push the branch to your forked repository: `git push origin feature/your-feature-name`
5. Open a pull request to the main repository.

Please ensure that your contributions align with the existing coding style and best practices.