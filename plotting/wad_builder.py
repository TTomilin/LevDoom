import subprocess as sp
from enum import Enum
from omg import *
from src.udmf import UDMFParser
from oblige import *

Monster = Enum('Monster', 'BaronOfHell, BetaSkull, Cacodemon, Cyberdemon, Demon, DoomImp, LostSoul, MarineBerserk, '
                          'MarineBFG, MarineChaingun, MarineChainsaw, MarineFist, MarinePistol, MarinePlasma, '
                          'MarineRailgun, MarineRocket, MarineShotgun, MarineSSG, MBFHelperDog, ScriptedMarine, '
                          'ShotgunGuy, Spectre, SpiderMastermind, StealthBaron, StealthCacodemon, StealthDemon, '
                          'StealthDoomImp, StealthShotgunGuy, StealthZombieMan, ZombieMan')
Weapon = Enum('Weapon', 'BFG9000, Chaingun, Chainsaw, Fist, Pistol, PlasmaRifle, RocketLauncher, Shotgun')
Ammo = Enum('Ammo', 'Backpack, Cell, CellPack, ClipBox, Clip, RocketAmmo, RocketBox, ShellBox, Shell')
textures = ['AASHITTY', 'AASTINKY', 'ASHWALL', 'ASHWALL2', 'ASHWALL3', 'ASHWALL4', 'ASHWALL6', 'ASHWALL7', 'BFALL1',
            'BFALL2', 'BFALL3', 'BFALL4', 'BIGBRIK1', 'BIGBRIK2', 'BIGBRIK3', 'BIGDOOR1', 'BIGDOOR1', 'BIGDOOR2',
            'BIGDOOR2', 'BIGDOOR3', 'BIGDOOR3', 'BIGDOOR4', 'BIGDOOR4', 'BIGDOOR5', 'BIGDOOR5', 'BIGDOOR6', 'BIGDOOR6',
            'BIGDOOR7', 'BIGDOOR7', 'BLAKWAL1', 'BLAKWAL2', 'BLODGR1', 'BLODGR2', 'BLODGR3', 'BLODGR4', 'BLODRIP1',
            'BLODRIP1', 'BLODRIP2', 'BLODRIP2', 'BLODRIP3', 'BLODRIP3', 'BLODRIP4', 'BLODRIP4', 'BRICK1', 'BRICK10',
            'BRICK11', 'BRICK12', 'BRICK2', 'BRICK3', 'BRICK4', 'BRICK5', 'BRICK6', 'BRICK7', 'BRICK8', 'BRICK9',
            'BRICKLIT', 'BRNBIGC', 'BRNBIGL', 'BRNBIGR', 'BRNPOIS', 'BRNPOIS', 'BRNPOIS2', 'BRNSMAL1', 'BRNSMAL1',
            'BRNSMAL2', 'BRNSMAL2', 'BRNSMALC', 'BRNSMALC', 'BRNSMALL', 'BRNSMALL', 'BRNSMALR', 'BRNSMALR', 'BRONZE1',
            'BRONZE2', 'BRONZE3', 'BRONZE4', 'BROVINE', 'BROVINE2', 'BROVINE2', 'BROWN1', 'BROWN1', 'BROWN144',
            'BROWN144', 'BROWN96', 'BROWN96', 'BROWNGRN', 'BROWNGRN', 'BROWNHUG', 'BROWNHUG', 'BROWNPIP', 'BROWNPIP',
            'BROWNWEL', 'BRWINDOW', 'BSTONE1', 'BSTONE2', 'BSTONE3', 'CEMENT1', 'CEMENT1', 'CEMENT2', 'CEMENT2',
            'CEMENT3', 'CEMENT3', 'CEMENT4', 'CEMENT4', 'CEMENT5', 'CEMENT5', 'CEMENT6', 'CEMENT6', 'CEMENT7',
            'CEMENT8', 'CEMENT9', 'CEMPOIS', 'COMP2', 'COMPBLUE', 'COMPBLUE', 'COMPOHSO', 'COMPSPAN', 'COMPSPAN',
            'COMPSTA1', 'COMPSTA1', 'COMPSTA2', 'COMPSTA2', 'COMPTALL', 'COMPTALL', 'COMPTILE', 'COMPUTE1', 'COMPUTE2',
            'COMPUTE3', 'COMPWERD', 'COMPWERD', 'CRACKLE2', 'CRACKLE4', 'CRATE1', 'CRATE1', 'CRATE2', 'CRATE2',
            'CRATE3', 'CRATELIT', 'CRATELIT', 'CRATINY', 'CRATINY', 'CRATWIDE', 'CRATWIDE', 'DBRAIN1', 'DBRAIN2',
            'DBRAIN3', 'DBRAIN4', 'DOOR1', 'DOOR1', 'DOOR3', 'DOOR3', 'DOORBLU', 'DOORBLU', 'DOORBLU2', 'DOORBLU2',
            'DOORHI', 'DOORRED', 'DOORRED', 'DOORRED2', 'DOORRED2', 'DOORSTOP', 'DOORSTOP', 'DOORTRAK', 'DOORTRAK',
            'DOORYEL', 'DOORYEL', 'DOORYEL2', 'DOORYEL2', 'EXITDOOR', 'EXITDOOR', 'EXITSIGN', 'EXITSIGN', 'EXITSTON',
            'EXITSTON', 'FIREBLU1', 'FIREBLU1', 'FIREBLU2', 'FIREBLU2', 'FIRELAV2', 'FIRELAV2', 'FIRELAV3', 'FIRELAV3',
            'FIRELAVA', 'FIRELAVA', 'FIREMAG1', 'FIREMAG1', 'FIREMAG2', 'FIREMAG2', 'FIREMAG3', 'FIREMAG3', 'FIREWALA',
            'FIREWALA', 'FIREWALB', 'FIREWALB', 'FIREWALL', 'FIREWALL', 'GRAY1', 'GRAY1', 'GRAY2', 'GRAY2', 'GRAY4',
            'GRAY4', 'GRAY5', 'GRAY5', 'GRAY7', 'GRAY7', 'GRAYBIG', 'GRAYBIG', 'GRAYDANG', 'GRAYPOIS', 'GRAYPOIS',
            'GRAYTALL', 'GRAYTALL', 'GRAYVINE', 'GRAYVINE', 'GSTFONT1', 'GSTFONT1', 'GSTFONT2', 'GSTFONT2', 'GSTFONT3',
            'GSTFONT3', 'GSTGARG', 'GSTGARG', 'GSTLION', 'GSTLION', 'GSTONE1', 'GSTONE1', 'GSTONE2', 'GSTONE2',
            'GSTSATYR', 'GSTSATYR', 'GSTVINE1', 'GSTVINE1', 'GSTVINE2', 'GSTVINE2', 'ICKDOOR1', 'ICKWALL1', 'ICKWALL1',
            'ICKWALL2', 'ICKWALL2', 'ICKWALL3', 'ICKWALL3', 'ICKWALL4', 'ICKWALL4', 'ICKWALL5', 'ICKWALL5', 'ICKWALL6',
            'ICKWALL7', 'ICKWALL7', 'LITE2', 'LITE3', 'LITE3', 'LITE4', 'LITE5', 'LITE5', 'LITE96', 'LITEBLU1',
            'LITEBLU1', 'LITEBLU2', 'LITEBLU3', 'LITEBLU4', 'LITEBLU4', 'LITEMET', 'LITERED', 'LITESTON', 'MARBFAC2',
            'MARBFAC2', 'MARBFAC3', 'MARBFAC3', 'MARBFAC4', 'MARBFACE', 'MARBFACE', 'MARBGRAY', 'MARBLE1', 'MARBLE1',
            'MARBLE2', 'MARBLE2', 'MARBLE3', 'MARBLE3', 'MARBLOD1', 'MARBLOD1', 'METAL', 'METAL', 'METAL1', 'METAL1',
            'METAL2', 'METAL3', 'METAL4', 'METAL5', 'METAL6', 'METAL7', 'MIDBARS1', 'MIDBARS3', 'MIDBRN1', 'MIDBRN1',
            'MIDBRONZ', 'MIDGRATE', 'MIDGRATE', 'MIDSPACE', 'MIDVINE1', 'MIDVINE2', 'MODWALL1', 'MODWALL2', 'MODWALL3',
            'MODWALL4', 'NUKE24', 'NUKE24', 'NUKEDGE1', 'NUKEDGE1', 'NUKEPOIS', 'NUKEPOIS', 'NUKESLAD', 'PANBLACK',
            'PANBLUE', 'PANBOOK', 'PANBORD1', 'PANBORD2', 'PANCASE1', 'PANCASE2', 'PANEL1', 'PANEL2', 'PANEL3',
            'PANEL4', 'PANEL5', 'PANEL6', 'PANEL7', 'PANEL8', 'PANEL9', 'PANRED', 'PIPE1', 'PIPE1', 'PIPE2', 'PIPE2',
            'PIPE4', 'PIPE4', 'PIPE6', 'PIPE6', 'PIPES', 'PIPEWAL1', 'PIPEWAL2', 'PLANET1', 'PLAT1', 'PLAT1', 'REDWALL',
            'REDWALL', 'REDWALL1', 'ROCK1', 'ROCK2', 'ROCK3', 'ROCK4', 'ROCK5', 'ROCKRED1', 'ROCKRED1', 'ROCKRED2',
            'ROCKRED2', 'ROCKRED3', 'ROCKRED3', 'SFALL1', 'SFALL2', 'SFALL3', 'SFALL4', 'SHAWN1', 'SHAWN1', 'SHAWN2',
            'SHAWN2', 'SHAWN3', 'SHAWN3', 'SILVER1', 'SILVER2', 'SILVER3', 'SK_LEFT', 'SK_RIGHT', 'SKIN2', 'SKIN2',
            'SKINBORD', 'SKINCUT', 'SKINCUT', 'SKINEDGE', 'SKINEDGE', 'SKINFACE', 'SKINFACE', 'SKINLOW', 'SKINLOW',
            'SKINMET1', 'SKINMET1', 'SKINMET2', 'SKINMET2', 'SKINSCAB', 'SKINSCAB', 'SKINSYMB', 'SKINSYMB', 'SKINTEK1',
            'SKINTEK2', 'SKSNAKE1', 'SKSNAKE1', 'SKSNAKE2', 'SKSNAKE2', 'SKSPINE1', 'SKSPINE1', 'SKSPINE2', 'SKSPINE2',
            'SKULWAL3', 'SKULWALL', 'SKY1', 'SKY1', 'SKY2', 'SKY2', 'SKY3', 'SKY3', 'SKY4', 'SLADPOIS', 'SLADPOIS',
            'SLADRIP1', 'SLADRIP2', 'SLADRIP3', 'SLADSKUL', 'SLADSKUL', 'SLADWALL', 'SLADWALL', 'SLOPPY1', 'SLOPPY2',
            'SP_DUDE1', 'SP_DUDE1', 'SP_DUDE2', 'SP_DUDE2', 'SP_DUDE3', 'SP_DUDE4', 'SP_DUDE4', 'SP_DUDE5', 'SP_DUDE5',
            'SP_DUDE6', 'SP_DUDE7', 'SP_DUDE8', 'SP_FACE1', 'SP_FACE1', 'SP_FACE2', 'SP_HOT1', 'SP_HOT1', 'SP_ROCK1',
            'SP_ROCK1', 'SP_ROCK2', 'SPACEW2', 'SPACEW3', 'SPACEW4', 'SPCDOOR1', 'SPCDOOR2', 'SPCDOOR3', 'SPCDOOR4',
            'STARBR2', 'STARBR2', 'STARG1', 'STARG1', 'STARG2', 'STARG2', 'STARG3', 'STARG3', 'STARGR1', 'STARGR1',
            'STARGR2', 'STARGR2', 'STARTAN1', 'STARTAN2', 'STARTAN2', 'STARTAN3', 'STARTAN3', 'STEP1', 'STEP1', 'STEP2',
            'STEP2', 'STEP3', 'STEP3', 'STEP4', 'STEP4', 'STEP5', 'STEP5', 'STEP6', 'STEP6', 'STEPLAD1', 'STEPLAD1',
            'STEPTOP', 'STEPTOP', 'STONE', 'STONE', 'STONE2', 'STONE2', 'STONE3', 'STONE3', 'STONE4', 'STONE5',
            'STONE6', 'STONE7', 'STONGARG', 'STONPOIS', 'STUCCO', 'STUCCO1', 'STUCCO2', 'STUCCO3', 'SUPPORT2',
            'SUPPORT2', 'SUPPORT3', 'SUPPORT3', 'SW1BLUE', 'SW1BLUE', 'SW1BRCOM', 'SW1BRCOM', 'SW1BRIK', 'SW1BRN1',
            'SW1BRN1', 'SW1BRN2', 'SW1BRN2', 'SW1BRNGN', 'SW1BRNGN', 'SW1BROWN', 'SW1BROWN', 'SW1CMT', 'SW1CMT',
            'SW1COMM', 'SW1COMM', 'SW1COMP', 'SW1COMP', 'SW1DIRT', 'SW1DIRT', 'SW1EXIT', 'SW1EXIT', 'SW1GARG',
            'SW1GARG', 'SW1GRAY', 'SW1GRAY', 'SW1GRAY1', 'SW1GRAY1', 'SW1GSTON', 'SW1GSTON', 'SW1HOT', 'SW1HOT',
            'SW1LION', 'SW1LION', 'SW1MARB', 'SW1MET2', 'SW1METAL', 'SW1METAL', 'SW1MOD1', 'SW1PANEL', 'SW1PIPE',
            'SW1PIPE', 'SW1ROCK', 'SW1SATYR', 'SW1SATYR', 'SW1SKIN', 'SW1SKIN', 'SW1SKULL', 'SW1SLAD', 'SW1SLAD',
            'SW1STARG', 'SW1STARG', 'SW1STON1', 'SW1STON1', 'SW1STON2', 'SW1STON2', 'SW1STON6', 'SW1STONE', 'SW1STONE',
            'SW1STRTN', 'SW1STRTN', 'SW1TEK', 'SW1VINE', 'SW1VINE', 'SW1WDMET', 'SW1WOOD', 'SW1WOOD', 'SW1ZIM',
            'SW2BLUE', 'SW2BLUE', 'SW2BRCOM', 'SW2BRCOM', 'SW2BRIK', 'SW2BRN1', 'SW2BRN1', 'SW2BRN2', 'SW2BRN2',
            'SW2BRNGN', 'SW2BRNGN', 'SW2BROWN', 'SW2BROWN', 'SW2CMT', 'SW2CMT', 'SW2COMM', 'SW2COMM', 'SW2COMP',
            'SW2COMP', 'SW2DIRT', 'SW2DIRT', 'SW2EXIT', 'SW2EXIT', 'SW2GARG', 'SW2GARG', 'SW2GRAY', 'SW2GRAY',
            'SW2GRAY1', 'SW2GRAY1', 'SW2GSTON', 'SW2GSTON', 'SW2HOT', 'SW2HOT', 'SW2LION', 'SW2LION', 'SW2MARB',
            'SW2MET2', 'SW2METAL', 'SW2METAL', 'SW2MOD1', 'SW2PANEL', 'SW2PIPE', 'SW2PIPE', 'SW2ROCK', 'SW2SATYR',
            'SW2SATYR', 'SW2SKIN', 'SW2SKIN', 'SW2SKULL', 'SW2SLAD', 'SW2SLAD', 'SW2STARG', 'SW2STARG', 'SW2STON1',
            'SW2STON1', 'SW2STON2', 'SW2STON2', 'SW2STON6', 'SW2STONE', 'SW2STONE', 'SW2STRTN', 'SW2STRTN', 'SW2TEK',
            'SW2VINE', 'SW2VINE', 'SW2WDMET', 'SW2WOOD', 'SW2WOOD', 'SW2ZIM', 'TANROCK2', 'TANROCK3', 'TANROCK4',
            'TANROCK5', 'TANROCK7', 'TANROCK8', 'TEKBRON1', 'TEKBRON2', 'TEKGREN1', 'TEKGREN2', 'TEKGREN3', 'TEKGREN4',
            'TEKGREN5', 'TEKLITE', 'TEKLITE2', 'TEKWALL1', 'TEKWALL1', 'TEKWALL2', 'TEKWALL3', 'TEKWALL4', 'TEKWALL4',
            'TEKWALL5', 'TEKWALL6', 'WOOD1', 'WOOD1', 'WOOD10', 'WOOD12', 'WOOD3', 'WOOD3', 'WOOD4', 'WOOD4', 'WOOD5',
            'WOOD5', 'WOOD6', 'WOOD7', 'WOOD8', 'WOOD9', 'WOODGARG', 'WOODGARG', 'WOODMET1', 'WOODMET2', 'WOODMET3',
            'WOODMET4', 'WOODSKUL', 'WOODVERT', 'ZDOORB1', 'ZDOORF1', 'ZELDOOR', 'ZIMMER1', 'ZIMMER2', 'ZIMMER3',
            'ZIMMER4', 'ZIMMER5', 'ZIMMER7', 'ZIMMER8', 'ZZWOLF1', 'ZZWOLF10', 'ZZWOLF11', 'ZZWOLF12', 'ZZWOLF13',
            'ZZWOLF2', 'ZZWOLF3', 'ZZWOLF4', 'ZZWOLF5', 'ZZWOLF6', 'ZZWOLF7', 'ZZWOLF9', 'ZZZFACE1', 'ZZZFACE2',
            'ZZZFACE3', 'ZZZFACE4', 'ZZZFACE5', 'ZZZFACE6', 'ZZZFACE7', 'ZZZFACE8', 'ZZZFACE9']


class WorldBuilder:

    def __init__(self, wad_path):
        self.wad_path = wad_path
        self.script_path = "../acc158win/SCRIPTS.acs"
        self.compiler_path = "../acc158win/acc.exe"
        self.behaviour_path = "../acc158win/BEHAVIOR.lmp"

    def swap_monster(self, monster1, monster2):
        wad_file = WAD(self.wad_path)
        wad_data = wad_file.data

        modified_script = wad_data['SCRIPTS'].data.decode("utf-8").replace(monster1.name, monster2.name)
        wad_data['SCRIPTS'].data = modified_script.encode()

        # Store the modified script for compilation
        with open(self.script_path, "w") as script:
            script.write(modified_script)

        # Compile the ACS
        sp.run([self.compiler_path, self.script_path, self.behaviour_path])

        # Read the compiled behaviour
        with open(self.behaviour_path, "rb") as behaviour:
            wad_data['BEHAVIOR'].data = behaviour.read()

        # Save the modified WAD
        wad_file.to_file(self.wad_path)

    def randomize_textures(self):
        wad_file = WAD(self.wad_path)
        wad_data = wad_file.data
        text_map = wad_data['TEXTMAP'].data.decode("utf-8")
        doom_map = UDMFParser.parse_textmap(text_map)
        sidedefs = doom_map['sidedefs']
        sidedefs[:] = [self.random_texture(sidedef) for sidedef in sidedefs]
        udmf_map = UDMFParser.convert_to_udmf(doom_map)
        bytes_map = bytes(udmf_map, 'utf-8')
        wad_data['TEXTMAP'].data = bytes_map
        wad_file.to_file(self.wad_path)

    def randomize_vertices(self):
        wad_file = WAD(self.wad_path)
        wad_data = wad_file.data
        text_map = wad_data['TEXTMAP'].data.decode("utf-8")
        doom_map = UDMFParser.parse_textmap(text_map)
        vertices = doom_map['vertices']
        vertices[:] = [random_vertex(vertex) for vertex in vertices]
        udmf_map = UDMFParser.convert_to_udmf(doom_map)
        bytes_map = bytes(udmf_map, 'utf-8')
        wad_data['TEXTMAP'].data = bytes_map
        wad_file.to_file(self.wad_path)

    def build_new_map(self):
        generator = DoomLevelGenerator()
        generator.set_config(map_config)
        new_map = generator.generate("../scenarios/new_map.wad")
        print('New Map', new_map)


def random_texture(element: dict) -> dict:
    element['texturemiddle'] = random.choice(textures)
    return element


def random_vertex(vertex: dict) -> dict:
    vertex['x'] = random.randrange(-800, 800)
    vertex['y'] = random.randrange(-800, 800)
    return vertex


map_config = {

    # General
    "game": "doom2",
    "engine": "vizdoom",
    "length": "single",
    "size": "micro",
    "caves": "none",
    "liquids": "none",
    "hallways": "none",
    "teleporters": "none",
    "steepness": "none",
    "mons": "nuts",
    "traps": "none",
    "cages": "none",
    "health": "more",
    "weapons": "sooner",

    # Modules
    "big_rooms": "heaps",
    "doors": "none",
    "keys": "none",
    "switches": "none",
}


# For testing
# wad_path = "../scenarios/defend_the_center_modified_textures.wad"
# wb = WorldBuilder(wad_path)
# wb.randomize_textures()
