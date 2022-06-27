# Additional Data On Google Drive
Some of the data is on google drive because it can't be uploaded
to GitHub due to the size of the folders.
In order to download the data please go to the following link: 
https://drive.google.com/drive/folders/1iQQRS1M6UXxEWdrIdlrwEivEIzvYcz9g

In the drive folder you will find some directories as zip files:
- ResizedImages.zip - Holds one folder contains all of the images in 150x150 fixed size.
- train.zip, test.zip, validation.zip - those zip files contain default train, test, validation directories splitted by 70,20,10. Those folders are the folders that the default model was trained on.
- model_data.zip - holds the default model.
- Images.zip - holds sample images for PoC of how we have resized the images.
    ```
    import data_organizer as dorg
    dorg.resize_images_on_directory()
    ```
    The following code will generate a sample ResizedImagesPoC folder that will holds the resized images.


## Important Notes

- Place all of the files within a new directory. The name of the new directory does not matter but for explanation we'll assume that it is "PythonProject".
- Unzip all of the files downloaded from the drive directly in the PythonProject folder.

You can follow the instruction in the book for diagrams of how the data should be placed exactly.