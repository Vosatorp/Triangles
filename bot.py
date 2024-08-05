import os
import pdb
import subprocess
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Define the command /start to initialize the bot
def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Hi! Send me a photo, and I will process it for you.')

# Define the function to handle photo messages
def handle_photo(update: Update, context: CallbackContext) -> None:
    # Get the photo file
    photo_file = update.message.photo[-1].get_file()
    photo_path = f'data/photo_{update.message.from_user.id}.jpg'
    photo_file.download(photo_path)

    update.message.reply_text("Starting processing...")

    # Run the find_numbers.py script
    test_dir = f'test_{update.message.from_user.id}'
    os.makedirs(test_dir, exist_ok=True)
    checkpoint = 'best.pt'  # Update this to the correct path if necessary

    conda_path = "/home/dsprotasov/miniconda3/bin/conda"
    conda_env = 'triangles'
    command = f'{conda_path} run -n {conda_env} python find_numbers.py --photos {photo_path} --test_dir {test_dir} --checkpoint {checkpoint}'
    # pdb.set_trace()
    # command = [
    #     'python', 'find_numbers.py',
    #     '--photos', photo_path,
    #     '--test_dir', test_dir,
    #     '--checkpoint', checkpoint
    # ]

    annotated_image_path = os.path.join(test_dir, 'annotated_image.png')
    result_txt_path = os.path.join(test_dir, 'result.txt')
    command_output_path = os.path.join(test_dir, 'command_output.txt')

    try:
        subprocess.run(command, shell=True, check=True)


        with open(command_output_path, 'w') as output_file:
            subprocess.run(command, shell=True, check=True, stdout=output_file, stderr=subprocess.STDOUT)

        # Send the annotated image back to the user as a file
        with open(annotated_image_path, 'rb') as annotated_image_file:
            update.message.reply_document(document=annotated_image_file)

        # Send the result.txt file back to the user
        with open(result_txt_path, 'rb') as result_txt_file:
            update.message.reply_document(document=result_txt_file)

        # Send the command output file back to the user
        with open(command_output_path, 'rb') as command_output_file:
            update.message.reply_document(document=command_output_file)

    except subprocess.CalledProcessError as e:
        update.message.reply_text('There was an error processing the image.')

    finally:
        # Clean up the photo and test directory
        os.remove(photo_path)
        if os.path.exists(annotated_image_path):
            os.remove(annotated_image_path)
        if os.path.exists(result_txt_path):
            os.remove(result_txt_path)
        os.rmdir(test_dir)

def main():
    # Create the Updater and pass it your bot's token
    updater = Updater(os.getenv("BOT_TOKEN_TR"), use_context=True)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # Register the /start command
    dispatcher.add_handler(CommandHandler("start", start))

    # Register the photo handler
    dispatcher.add_handler(MessageHandler(Filters.photo, handle_photo))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT, SIGTERM or SIGABRT
    updater.idle()

if __name__ == '__main__':
    main()
