import argparse
import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, F, Router
from aiogram.types import Message
from faster_whisper import WhisperModel


logging.basicConfig(level=logging.INFO, format="%(name)s:%(levelname)s:%(asctime)s:%(message)s")


r = Router(name="whisper")


def parse_cmd_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--local",
        action="store_true",
        help="run pre-downloaded model",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=".",
        help="path to the model (pre-downloaded or to download to)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="logging level",
    )
    args = parser.parse_args()
    return args


@r.message(F.voice)
async def handle_voice_message(message: Message, bot: Bot, whisper_model: WhisperModel):
    logger = logging.getLogger(__name__)
    status_msg = await message.answer("🎙 Volice message received. Processing...")

    if message.voice is None:
        logging.warning("no voice message was found")
        status_msg.edit_text("Strange, but voice message is missing 🤔\n"+
                                "Please, try again 🔁")
        return
    
    file_id = message.voice.file_id
    ogg_file = f"voice_{file_id}.ogg"
    
    try:
        file_info = await bot.get_file(message.voice.file_id)
        logger.debug(f'saving file to {file_info.file_path}')
        file_path = file_info.file_path
        if file_path is None:
            logger.warning("file_path is empty, can't download file")
            status_msg.edit_text("Something went wrong 😔")
            return
        
        await bot.download_file(file_path, ogg_file)

        await status_msg.edit_text("⏳ Recognizing speech (STT)...")
        
        logger.debug('running whisper')
        loop = asyncio.get_running_loop()
        segments, _ = await loop.run_in_executor(
            None, 
            lambda: whisper_model.transcribe(ogg_file, vad_filter=True)
        )

        user_text = "".join([segment.text for segment in segments]).strip()
        
        if not user_text:
            await status_msg.edit_text("❌ Couldn't make out the words. "+
                                       "Please, try to speak more clearly 🙏")
            return

        await status_msg.edit_text(f"🗣 **Voice message content:** _{user_text}_\n\n", parse_mode="Markdown")
                
    except Exception as e:
        logger.error(f"failed to handle: {e}")
        await status_msg.edit_text("💥 Oops! An error occurred while processing the message.")
        
    finally:
        # Удаляем временный файл с диска
        if os.path.exists(ogg_file):
            os.remove(ogg_file)


@r.message(F.text)
async def handle_text_message(message: Message):
    await message.answer("🗣️ Would you kindly send me a voice message?")

async def main():
    args = parse_cmd_args()
    logger = logging.getLogger(__name__)
    log_level: str = args.log_level
    logger.setLevel(log_level.upper())
    logger.info("Loading Whisper model...")
    whisper_model = WhisperModel(
        "medium", # using medium model
        device="auto", # automatically choosing device to run on
        compute_type="float16",
        download_root=args.path, # download model locally
        local_files_only=args.local, # allow to load Faster Whisper from Hugging Face
    )

    BOT_TOKEN = os.getenv("TG_WHISPER_BOT_TOKEN") or ""
    bot = Bot(token=BOT_TOKEN)

    dp = Dispatcher()
    dp.include_router(router=r)
    logger.info("bot started")
    await dp.start_polling(
        bot,
        whisper_model=whisper_model,
    )

if __name__ == "__main__":
    asyncio.run(main())
