from tkinter import *
from tkinter.filedialog import askopenfilenames
from keras.models import load_model
from threading import Thread
from model import train_model, evaluate_model, draw_model_graph
import numpy as np
import data_organizer as _dorg
import tkinter.messagebox as mbox
import cv2
import os


# holds the functions of the main menu of the application.
class MainMenuForm:
    model_op = None  # the model that we are testing on.

    def __init__(self):
        self.main_app = None  # the main app loop
        self.app_title = None  # the title of the main menu of the app.
        self.select_model_btn = None  # the button that represents the Select model.
        # this button is responsible for asking the user if he would rather create his own model
        # or use the already existed model.

        self.test_chosen_model_btn = None  # the button which is responsible for testing the model.
        # the button opens a form that gives the user multiple options to test the chosen model.

    # the main menu of the app.
    def start_main_menu(self):
        # the main app loop
        self.main_app = Tk()

        # the title of the main menu of the app.
        self.app_title = Label(self.main_app,
                               text='Mask Classification Program!')
        self.app_title.grid(row=0, column=0)

        # the button that represents the Select model.
        # this button is responsible for asking the user if he would rather create his own model
        # or use the already existed model.
        self.select_model_btn = Button(self.main_app,
                                       text='Select Model',
                                       command=self._ask_user_to_select_model)
        self.select_model_btn.grid(row=1, column=0, padx=10, pady=10)

        # the button which is responsible for testing the model.
        # the button opens a form that gives the user multiple options to test the chosen model.
        self.test_chosen_model_btn = Button(self.main_app, text='Test Model', state=DISABLED,
                                            command=self._start_testing_model)
        self.test_chosen_model_btn.grid(row=2, column=0, padx=10, pady=10)

        self.main_app.mainloop()

    # this function starts the main testing model menu.
    def _start_testing_model(self):
        testing_form = TestModelForm()
        testing_form.start_main_test_menu()

    # managing user decision about what model to choose.
    # test_chosen_model_btn -> we unlock the button if we have a model to test on.
    def _ask_user_to_select_model(self):
        # we only do this in case that there is a default model.
        if os.path.exists('model_data/'):
            # responsible for questioning the user if he would rather train the model again or use the default one.
            q_res = mbox.askyesno(title='Model Already exists!',
                                        message='It seems like there is an already existed model.'
                                                'Would you rather use the default model?')
            # the user agrees to use the default model.
            if q_res:
                MainMenuForm.model_op = load_model('model_data')  # we load the already existed model.
                self.test_chosen_model_btn['state'] = NORMAL  # we set the next button to be normal because now when we
                # have a model there is no reason that he would not be intractable.
                self.select_model_btn['state'] = DISABLED
                return

        self.select_model_btn['state'] = DISABLED
        # we start the form that responsible for building the model.
        _build_model_app = BuildModelForm(self)
        _build_model_app.start_build_model_menu()


# holds the functions of the model building part of the application.
class BuildModelForm:

    tempModel = None  # the model we are creating in the current form.

    def __init__(self, main_menu_form: MainMenuForm):
        self.m_Menu = main_menu_form  # holds a reference to the main menu form. We need that in order to make
        # the testing model button available after the training process.
        self.is_form_closed = False  # indicates if the form is closed or not.
        self._build_model_app = None  # the build model app loop
        self._build_model_title = None  # the title of the building model app.

        self._organize_data_set_btn = None  # the button which is responsible for organizing the data.
        # organizing the data means splitting the data into train, test, validation folders
        # using the data_organizer.py module.

        self._split_entry = None  # responsible for asking the user how he would like to split the
        # train, test, validation.

        self._steps_split = [70, 20, 10]  # holds a list of the train, test, validation percent split.

        self._send_split_entry_btn = None  # sends the data on the split entry to the train, test, validation split.
        self._train_model_btn = None  # responsible for training the new model.
        self._close_btn = None  # closing the form.

    # the menu where we are building the model.
    def start_build_model_menu(self):
        # the build model app loop
        self._build_model_app = Tk()

        # the title of the building model app.
        self._build_model_title = Label(self._build_model_app, text='Build Your Model:')
        self._build_model_title.grid(row=0, column=0, padx=10, pady=10)

        # the button which is responsible for organizing the data.
        # organizing the data means splitting the data into train, test, validation folders
        # using the data_organizer.py module.
        self._organize_data_set_btn = Button(self._build_model_app, text='Organize Dataset',
                                             command=self._ask_user_if_split_exists_already)
        self._organize_data_set_btn.grid(row=1, column=0, padx=10, pady=10)

        # responsible for training the new model.
        self._train_model_btn = Button(self._build_model_app, text='Train Model', state=DISABLED,
                                       command=self._start_training_model)
        self._train_model_btn.grid(row=4, column=0, padx=10, pady=10)

        # closing the form.
        self._close_btn = Button(self._build_model_app, text='Main Menu', command=self.onclose)
        self._close_btn.grid(row=5, column=0, padx=10, pady=10)

        self._build_model_app.protocol("WM_DELETE_WINDOW", self.onclose)
        self._build_model_app.mainloop()

    # a function with a responsibility to train the new model.
    def _th_train_model(self):
        self._train_model_btn['state'] = DISABLED
        _model = train_model()
        draw_model_graph()

        if self.is_form_closed:  # if the form is closed we don't update anything.
            return

        BuildModelForm.tempModel = _model
        self.m_Menu.test_chosen_model_btn['state'] = NORMAL  # we set the testing model choice to True.
        mbox.showinfo(title='End Train', message='Success!')

    # function responsibility is to train the model.
    # the only responsibility of the function is to call the train model function.
    # the reason why we use the function and not put _th_train_model right away
    # is that we don't want to hold the command.
    # if the command in the button event takes too much time to complete the tkinter app crashes. that's why we use
    # a different thread and instead we just don't allow to continue because the next buttons are all unavailable until
    # the task is completed. so we don't even use thread.join().
    def _start_training_model(self):
        mbox.showinfo(title='Start Train',
                      message='We are about to start training, if you would like to watch the process of the app'
                              'please look at the command line/IDE terminal'
                              ' from which you have opened the tkinter app from\n'
                              'When the process will over a Success message will be shown.')

        train_thread = Thread(target=self._th_train_model)
        train_thread.start()

    # we ask the user if he wants to re-split the data to train, test, validation folders if the data
    # exists there already.
    # organize_btn -> we disable the organize btn so no change will occur during asking for input.
    def _ask_user_if_split_exists_already(self):
        cwd_directories = os.listdir(os.getcwd())  # we get the names of all directories in the current directory.

        # we check if the train, test, validation already exists.
        # if they does we ask the user if he would like to override them.
        if all([x in cwd_directories for x in ['train', 'test', 'validation']]):
            # responsible for questioning the user if he would like to re-organize the data.
            q_res = mbox.askyesno(title='Would you rather reorganize?',
                                  message='It seems like you already have an organized data.'
                                          'Would you rather to override the existed directories?')

            # means that the user does not want to override the existing context.
            if not q_res:
                self._train_model_btn['state'] = NORMAL  # we set the train model button to be enabled.
                self._organize_data_set_btn['state'] = DISABLED
                return

        # we re build the directories.
        mbox.showinfo(title='Split Data',
                      message='Please choose the way that you would like to split your data in. '
                      'If you enter nothing and press submit the data split will occur in the '
                      'following way. train=70%, test=20%, validation=10%. '
                      'If you prefer to split the data in your own way please write it in the '
                      'following format x,y,z for example 70,20,10. The first goes for train '
                      'the second goes for test and the third goes for validation. '
                      'Please enter 3 numbers that their sum is = 100 unless an exception '
                      'will be thrown. ')

        # we ask the user how he would like to split the directories.
        self._split_entry = Entry(self._build_model_app)
        self._split_entry.grid(row=2, column=0, padx=10, pady=10)

        # we add a button to have the option to send the data.
        self._send_split_entry_btn = Button(self._build_model_app, text='Submit Split',
                                            command=self._validate_input_for_steps)
        self._send_split_entry_btn.grid(row=3, column=0, padx=10, pady=10)

    # the function is responsible for handling the data_organizing process.
    def _th_start_organizing_the_data(self):

        # we alert the user that the organizing process began.
        mbox.showinfo(title='Start Organizing',
                      message='We are about to start organizing, if you would like to watch the process of the app'
                              'please look at the command line/IDE terminal '
                              'from which you have opened the tkinter app from.\n'
                              'When the process will over a Success message will be shown.')

        self._send_split_entry_btn['state'] = DISABLED
        self._organize_data_set_btn['state'] = DISABLED

        # we call the functions that are responsible for organizing the data.
        _dorg.build_directories()
        _dorg.divide_to_train_test_validation(*[x / 100.0 for x in self._steps_split])

        # if we have passed all cases it means that we have a valid input and we can set the splitting to what
        # we have got.
        self._train_model_btn['state'] = NORMAL  # we set the next button to be available.

        mbox.showinfo(title='End Organizing', message='Success!')

    # the function validates the input from the split entry field.
    def _validate_input_for_steps(self):
        user_input = self._split_entry.get()
        temp_split_format = []

        # the user did not enter any splitting way.
        if user_input.strip() == '':
            # we call the functions that are responsible for organizing the data.
            # we start the organizing process.
            _data_organizing_thread = Thread(target=self._th_start_organizing_the_data)
            _data_organizing_thread.start()
            return  # we will use the default [70, 20, 10]

        user_input_list = user_input.split(',')  # we split the user input by comma.

        # if we did not receive 3 values it means that the input was not formatted well.
        if len(user_input_list) != 3:
            mbox.showerror(title='Splitting Error', message=f'Program expected to have 3 inputs but '
                                                            f'{len(user_input_list)} were given.')
            return

        # we validate that all inputs can be parsed as integers.
        try:
            for x in user_input_list:
                temp_split_format.append(int(x.strip()))

        except ValueError:
            mbox.showerror(title='Splitting Error', message='One or more of your inputs were not integers.')
            return

        # if the user tried to enter negetive number
        if not all([x > 0 for x in temp_split_format]):
            mbox.showerror(title='Splitting Error', message='One or more of your inputs were a negetive number')
            return

        # we check that the inputs we have got does not go over 100.
        if sum(temp_split_format) != 100:
            mbox.showerror(title='Splitting Error', message='You have entered values with sum different than 100%.')
            return
        self._steps_split = temp_split_format  # we set the split to the split we got from the user.

        # we start the organizing process.
        _data_organizing_thread = Thread(target=self._th_start_organizing_the_data)
        _data_organizing_thread.start()

    # on close we
    def onclose(self):
        self.is_form_closed = True  # the form is closed.

        if BuildModelForm.tempModel is None:
            self.m_Menu.select_model_btn['state'] = NORMAL

        self._build_model_app.destroy()

# holds the functions of the testing model part of the application.
class TestModelForm:

    def __init__(self):
        # if we got here then we must have at least one model ready (unless the test model button were not available).
        # we check which one of them is valid. If tempModel is not None it means we have trained the model and we
        # did not use the default one.
        self.model = BuildModelForm.tempModel if BuildModelForm.tempModel is not None else MainMenuForm.model_op

        self.test_title = None  # the title of the app.
        self.test_menu = None  # the main test menu app.
        self.evaluate_by_test_folder_btn = None  # testing with the test folder (which we have on disk).
        self.test_with_new_pictures_btn = None  # the button lets the user to pick his own pictures for the test.

    # we start the main test menu.
    # from this menu we can choose multiple ways to test our model.
    def start_main_test_menu(self):
        self.test_menu = Tk()

        self.test_title = Label(self.test_menu, text='Test Your Model: ')
        self.test_title.grid(row=0, column=0, padx=10, pady=10)

        self.evaluate_by_test_folder_btn = Button(self.test_menu, text='Evaluate Model by Test Folder',
                                                  command=self.show_model_evaluation)
        self.evaluate_by_test_folder_btn.grid(row=1, column=0, padx=10, pady=10)

        self.test_with_new_pictures_btn = Button(self.test_menu, text='Test With Your Own Pictures',
                                                 command=TestWithPicturesForm(self.model).open_images_for_test)
        self.test_with_new_pictures_btn.grid(row=2, column=0, padx=10, pady=10)

        self.test_menu.mainloop()

    # a thread responsible for evaluating the model and giving a message after.
    def _th_evaluate_model(self):
        print('\n\n')
        res = evaluate_model(self.model)
        mbox.showinfo(title='Model Testing Results',
                      message=f'The model finished training with the following results.\n'
                              f'{res}')

    # function responsibility is to evaluate the model.
    # the only responsibility of the function is to call the evaluate model function.
    # the reason why we use the function and not put _th_evaluate_model
    #  right away is that we don't want to hold the command.
    # if the command in the button event takes too much time to complete the tkinter app crashes.
    def show_model_evaluation(self):
        mbox_th = Thread(target=lambda: mbox.showinfo(title='Evaluating...', message= 'Evaluating... '
                                                       'Wait for a message that shows the results.'
                                                       'If you want to see the process occurs you can view it '
                                                       'with the command line / IDE terminal.'))

        mbox_th.start()

        print('Starting Testing...')
        evaluate_model_thread = Thread(target=self._th_evaluate_model)
        evaluate_model_thread.start()


# holds the functions related to testing the application with new images.
class TestWithPicturesForm:

    def __init__(self, model):
        self._model = model  # the model used for test.

    # called first to select the images the user wants to pick.
    def open_images_for_test(self):
        file_names = askopenfilenames()  # the file name we are willing to use.

        # we first validate our inputs.
        for _file_name in file_names:
            f_ext = os.path.splitext(_file_name)[1]  # the file extension
            if f_ext.lower() not in ['.jpg', '.png', '.jpeg']:
                mbox.showerror(title='Invalid Input',
                               message=f'.jpg was expected but {f_ext} was the input.')
                return

        for _name in file_names:
            cv2_img = cv2.imread(_name)
            cv2_img = cv2.resize(cv2_img, (150, 150))
            cv2_img = cv2_img / 255
            predicted_label = TestWithPicturesForm.predict(self._model, cv2_img)
            cv2.putText(img=cv2_img, text=predicted_label, org=(0, 30), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=0.8, color=(0, 255, 0), thickness=1)

            cv2.imshow(_name, cv2_img)
            cv2.waitKey(0)

    @staticmethod
    def predict(model, img):
        pred = model.predict(img[None, ...])
        pred = np.argmax(pred, axis=1)

        if pred == 0:
            return 'Full'  # the mask is on the nose
        elif pred == 1:
            return 'Half'  # the mask in on the mouth
        elif pred == 2:
            return 'No'  # there is no mask.
        else:
            return 'Bottom'  # the mask is on the chin.


if __name__ == '__main__':
    main = MainMenuForm()
    main.start_main_menu()

