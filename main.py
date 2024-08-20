import subprocess
from getpass import getpass
from subprocess import Popen, PIPE
def run_command(command):
    print("Enter command")
    """Run a shell command and print its output."""
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    
    if result.returncode == 0:
        print(f"Successfully ran: {command}\nOutput:\n{result.stdout}")
    else:
        print(f"Error running: {command}\nError:\n{result.stderr}")
        exit(1)  # Exit if a command fails

def main():
    # List of commands to run
    commands = [
        "singularity run main.sif -p initialise",
        "singularity exec main.sif python sastvd/scripts/prepare.py",
        "singularity exec main.sif python sastvd/scripts/train_best.py"
    ]


    sudo_password = '0123456789'
    command = 'sudo singularity build main.sif Singularity'
    command = command.split()

    cmd1 = subprocess.Popen(['echo',sudo_password], stdout=subprocess.PIPE)
    cmd2 = subprocess.Popen(['sudo','-S'] + command, stdin=cmd1.stdout, stdout=subprocess.PIPE)

    output = cmd2.stdout.read().decode() 
    print(output)
    # Run each command in sequence
    for cmd in commands:
        run_command(cmd)

if __name__ == "__main__":
    main()
