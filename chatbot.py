from hugchat import hugchat
from hugchat.login import Login

sign = Login("TheOneAboveAll511", "Akif@2006")
cookies = sign.login()

cookie_path_dir = "./cookies_snapshot"
sign.saveCookiesToDir(cookie_path_dir)

chatbot = hugchat.ChatBot(cookies=cookies.get_dict())

id = chatbot.new_conversation()
chatbot.change_conversation(id)

msg = input("Enter your prompt here:")
print(chatbot.chat(msg))
