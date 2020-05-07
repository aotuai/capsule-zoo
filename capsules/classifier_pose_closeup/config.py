from vcap import FloatOption

pose_confidence = "pose_confidence"
pose_iou = "pose_iou"
pose = "Pose"

ground_poses = [
    "fall down",
    "get up",
    "crawl",
    "bend/bow (at the waist)",
    "lie/sleep",
]

standing_poses = [
    "run/jog",
    "stand",
    "walk",
    "jump/leap",

]

sitting_poses = [
    "sit",
    "crouch/kneel",
]

all_poses = ground_poses + sitting_poses + standing_poses + ["unknown"]

ignore = [
    "answer phone",
    "swim",
    "brush teeth",
    "martial art",
    "carry/hold (an object)",
    "catch (an object)",
    "clink glass",
    "close (e.g., a door, a box)",
    "cook",
    "cut",
    "dig",
    "dance",
    "chop",
    "climb (e.g., a mountain)",
    "dress/put on clothing",
    "drink",
    "drive (e.g., a car, a truck)",
    "eat",
    "enter",
    "exit",
    "extract",
    "fishing",
    "hit (an object)",
    "kick (an object)",
    "lift/pick up",
    "listen (e.g., to music)",
    "open (e.g., a window, a car door)",
    "paint",
    "play board game",
    "play musical instrument",
    "play with pets",
    "point to (an object)",
    "press",
    "pull (an object)",
    "push (an object)",
    "put down",
    "read",
    "ride (e.g., a bike, a car, a horse)",
    "row boat",
    "sail boat",
    "shoot",
    "shovel",
    "smoke",
    "stir",
    "take a photo",
    "text on/look at a cellphone",
    "throw",
    "touch (an object)",
    "turn (e.g., a screwdriver)",
    "watch (e.g., TV)",
    "work on a computer",
    "write",
    "fight/hit (a person)",
    "give/serve (an object) to (a person)",
    "grab (a person)",
    "hand clap",
    "hand shake",
    "hand wave",
    "hug (a person)",
    "kick (a person)",
    "kiss (a person)",
    "lift (a person)",
    "listen to (a person)",
    "play with kids",
    "push (another person)",
    "sing to (e.g., self, a person, a group)",
    "take (an object) from (a person)",
    "talk to (e.g., self, a person, a group)",
    "watch (a person)"
]

confidence_threshold = "confidence_threshold"
iou_threshold = "iou_threshold"
capsule_options = {
    confidence_threshold: FloatOption(
        default=0.5,
        min_val=0.0,
        max_val=1.0),
    iou_threshold: FloatOption(
        default=0.5,
        min_val=0.0,
        max_val=1.0
    )
}
