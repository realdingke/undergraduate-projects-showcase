import random

def name_to_number(name_input):
    """This function takes string inputs: rock-Spock-paper-lizard-scissors
    returns 0-1-2-3-4."""
    if name_input=="rock":
        return 0
    elif name_input=="Spock":
        return 1
    elif name_input=="paper":
        return 2
    elif name_input=="lizard":
        return 3
    elif name_input=="scissors":
        return 4
    else:
        return "invalid option, there is an error in the player's choice!"
    pass


def number_to_name(number_input):
    """This function takes number inputs in range of 0 to 4 inclusive,
    and returns the corresponding names as strings."""
    if number_input==0:
        return "rock"
    elif number_input==1:
        return "Spock"
    elif number_input==2:
        return "paper"
    elif number_input==3:
        return "lizard"
    elif number_input==4:
        return "scissors"
    else:
        return "invalid option, there is an error in the computer's choice!!"
    pass


def rpsls(player_choice):
    """This function takes the player's choice,
    prints put the player's choice,
    prints out the computer's choice,
    and prints out the winner of the game."""
    print("Player chooses "+str(player_choice)+".")
    player_number=name_to_number(player_choice);
    
    # The next line generates the computer's guess.
    comp_number=random.randint(0,4);
    comp_choice=number_to_name(comp_number);
    print("Computer chooses "+str(comp_choice)+".")
    
    # To print a warning against invalid player choice
    if player_number=="invalid option, there is an error in the player's choice!":
        print("The player has just chosen an invalid option!")       
    else:
        # The next line calculates the difference between comp and player, and then mirrors the negative diff to the positive axis, using diff=0 as the symmetry. 
        diff=(comp_number-player_number);
        # Next is to use conditionals to determine the winner.
        if (diff==-4) or (diff==-3) or (diff==1) or (diff==2):
            print("Computer wins!")
        elif (diff==-2) or (diff==-1) or (diff==3) or (diff==4):
            print("Player wins!")
        elif diff==0:
            print("A draw!")
        else:
            print("Sorry, we cannot judge becasue something went wrong.")
    
    print("")
    pass


# This section is for testing only.
rpsls("rock")
rpsls("Spock")
rpsls("scissors")
rpsls("sb")
rpsls("paper")
rpsls("sb")
