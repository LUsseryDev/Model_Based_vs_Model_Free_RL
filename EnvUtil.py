import gymnasium as gym
import ale_py

def getGameList() -> list[str]:
    gameList = [
    "Alien",
    "Amidar",
    "Assault",
    "Atlantis",
    "Asteroids",
    "Asterix",
    "BattleZone",
    "BankHeist",
    "Breakout",
    "Boxing",
    "Berzerk",
    "BeamRider",
    "ChopperCommand",
    "Centipede",
    "DemonAttack",
    "Defender",
    "CrazyClimber",
    "Enduro",
    "DoubleDunk",
    "FishingDerby",
    "Gravitar",
    "Gopher",
    "Frostbite",
    "Freeway",
    "Jamesbond",
    "IceHockey",
    "Hero",
    "KungFuMaster",
    "Krull",
    "Kangaroo",
    "NameThisGame",
    "MsPacman",
    "MontezumaRevenge",
    "PrivateEye",
    "Pong",
    "Pitfall",
    "Phoenix",
    "Robotank",
    "RoadRunner",
    "Riverraid",
    "Qbert",
    "SpaceInvaders",
    "Solaris",
    "Skiing",
    "Seaquest",
    "Surround",
    "StarGunner",
    "Tutankham",
    "TimePilot",
    "Tennis",
    "WizardOfWor",
    "VideoPinball",
    "UpNDown",
    "Zaxxon",
    "YarsRevenge"]

    return gameList

def getEnvList(gameList:list[str]) -> list[gym.Env]:
    envList = []

    #list.append(gym.make("ALE/Alien-v5", obs_type="grayscale", frameskip=3, full_action_space=True))

    for game in gameList:
        envList.append(gym.make("ALE/"+game+"-v5", obs_type="grayscale", frameskip=1))

    return envList

def chooseGames() -> list[str]:
    while True:
        print("1. use all environments")
        print("2. use some environments")
        print("3. use one environment")

        choice = input("Enter your choice: ")
        fullGameList = getGameList()
        match choice:
            case "1":
                return fullGameList
            case "2":
                tempGameList = []
                while True:

                    game = input("Enter game or -1 to finish list: ")
                    if game == "-1":
                        print("final game list:")
                        for g in tempGameList:
                            print(g)
                        return tempGameList
                    if game in fullGameList:
                        tempGameList.append(game)
                        print("current game list:")
                        for g in tempGameList:
                            print(g)
                    else:
                        print("Invalid game, please try again")

            case "3":
                while True:
                    game = input("Enter game: ")
                    if game in fullGameList:
                        print("selected game:")
                        print(game)
                        return [game]
                    else:
                        print("Invalid game, please try again")

            case _:
                print("invalid choice, please try again")