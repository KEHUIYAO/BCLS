import paramiko
from paramiko import SSHException


from getpass import getpass

from threading import Thread
import time
import sys


if __name__ == "__main__":

    #
    ssh = paramiko.SSHClient()
    #
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())


    def my_handler(title, instructions, prompt_list):
        answers = []
        for prompt_, echo in prompt_list:
            prompt = prompt_.strip().lower()

            if 'password' in prompt:
                answers.append("{Ray19960527}")
            elif 'duo' in prompt:
                duo = input('duo password:')
                answers.append(duo)
            else:
                answers.append(echo or getpass(prompt))
        return answers


        # print(title)
        # print(instructions)
        # return [echo and input(prompt) or getpass(prompt) for (prompt, echo) in prompt_list]


    try:
        ssh.connect('lunchbox.stat.wisc.edu', username='kehui')
    except paramiko.ssh_exception.SSHException:
        pass

    ssh.get_transport().auth_interactive(username='kehui', handler=my_handler)
    # ssh.connect('lunchbox.stat.wisc.edu', 22, 'kehui', '{Ray19960527}')

    channel = ssh.invoke_shell()

    # do some routine
    channel_data = str()
    while True:
        if channel.recv_ready():
            channel_data += channel.recv(9999).decode('utf-8')
        else:
            continue


        if channel_data.endswith('kehui@lunchbox:~$ '):
            print(channel_data)
            channel_data = str()
            channel.send('ssh kyao24@submit2.chtc.wisc.edu\n')

        elif channel_data.endswith('Password: '):
            print(channel_data)
            channel_data = str()
            channel.send('{Ray19960527}\n')

        elif channel_data.endswith('[kyao24@submit2 ~]$ '):
            print(channel_data)
            channel_data = str()
            channel.send('cd template-soil-moisture-project/soil_moisture_project/\n')

        elif channel_data.endswith('[kyao24@submit2 soil_moisture_project]$ '):
            print(channel_data)
            channel_data = str()
            #channel.send('git pull origin master\n')
            channel.send('cd conda\n')
            #channel.send('condor_q\n')

        elif channel_data.endswith('[kyao24@submit2 conda]$ '):
            print(channel_data)
            channel_data = str()
            channel.send('\n')

            def my_forever_while():
                "interact with the server every 5 seconds, so that the server will never time out you"
                global thread_running
                start_time = time.time()


                while thread_running:
                    time.sleep(0.1)
                    if time.time() - start_time >= 5:
                        start_time = time.time()
                        channel.send('\n')
                        #print("please do not time out!")

                    # receive data, otherwise, the buffer will blow up!
                    if channel.recv_ready():
                        channel.recv(9999).decode('utf-8')
                    else:
                        continue


            def take_input_1():
                "listen to user's command to interact with the server"
                global thread_running
                global switch

                mycommand = input('>') + '\n'
                #print(mycommand)
                if mycommand != "q\n":
                    channel.send(mycommand)
                else:
                    switch = 1


            def take_input_2():
                "listen to user's input, switch to user's control mode"
                global switch
                mode = input('waiting for you, the session never dies :)')
                if mode == '1':
                    switch = 3
                else:
                    switch = 2




            def multi_threading(channel_data):
                "Core function for switching back and forth from user's control mode and waiting mode"
                global thread_running
                global switch
                switch = 1

                while True:
                    if switch == 1:
                        thread_running = True
                        t1 = Thread(target=my_forever_while)
                        t2 = Thread(target=take_input_2)
                        t1.start()
                        t2.start()
                        t2.join()

                        thread_running = False
                        channel.send('\n')

                    elif switch == 2:
                        if channel.recv_ready():
                            channel_data += channel.recv(9999).decode('utf-8')
                        else:
                            continue

                        if 'kyao24@submit2' in channel_data:
                            print(channel_data)
                            channel_data = str()
                            take_input_1()

                    elif switch == 3:
                        if channel.recv_ready():
                            channel_data += channel.recv(9999).decode('utf-8')
                        else:
                            continue

                        if channel_data:
                            print(channel_data, end='', sep='')
                            channel_data = str()
                            take_input_1()

            multi_threading(channel_data)

        # elif 'timed out' in channel_data:
        #     channel.aend('\n')
        #     break




    # ssh.close()











