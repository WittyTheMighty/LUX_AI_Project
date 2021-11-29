import luxai2021.game.actions as luxai_actions
import luxai2021.game.constants as luxai_constants
import numpy as np
import sys
import time

from functools import partial
from gym import spaces
from luxai2021.env.agent import Agent, AgentWithModel
from luxai2021.game.game_constants import GAME_CONSTANTS
from luxai2021.game.position import Position


def closest_node(node, nodes):
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmin(dist_2)


def furthest_node(node, nodes):
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmax(dist_2)


def smart_transfer_to_nearby(game, team, unit_id, unit, target_type_restriction=None, **kwarg):
    """
    Smart-transfers from the specified unit to a nearby neighbor. Prioritizes any
    nearby carts first, then any worker. Transfers the resource type which the unit
    has most of. Picks which cart/worker based on choosing a target that is most-full
    but able to take the most amount of resources.

    Args:
        team ([type]): [description]
        unit_id ([type]): [description]

    Returns:
        Action: Returns a TransferAction object, even if the request is an invalid
                transfer. Use TransferAction.is_valid() to check validity.
    """

    # Calculate how much resources could at-most be transferred
    resource_type = None
    resource_amount = 0
    target_unit = None

    if unit is not None:
        for type, amount in unit.cargo.items():
            if amount > resource_amount:
                resource_type = type
                resource_amount = amount

        # Find the best nearby unit to transfer to
        unit_cell = game.map.get_cell_by_pos(unit.pos)
        adjacent_cells = game.map.get_adjacent_cells(unit_cell)

        for c in adjacent_cells:
            for id, u in c.units.items():
                # Apply the unit type target restriction
                if target_type_restriction is None or u.type == target_type_restriction:
                    if u.team == team:
                        # This unit belongs to our team, set it as the winning transfer target
                        # if it's the best match.
                        if target_unit is None:
                            target_unit = u
                        else:
                            # Compare this unit to the existing target
                            if target_unit.type == u.type:
                                # Transfer to the target with the least capacity, but can accept
                                # all of our resources
                                if (u.get_cargo_space_left() >= resource_amount and
                                        target_unit.get_cargo_space_left() >= resource_amount):
                                    # Both units can accept all our resources. Prioritize one that is most-full.
                                    if u.get_cargo_space_left() < target_unit.get_cargo_space_left():
                                        # This new target it better, it has less space left and can take all our
                                        # resources
                                        target_unit = u

                                elif target_unit.get_cargo_space_left() >= resource_amount:
                                    # Don't change targets. Current one is best since it can take all
                                    # the resources, but new target can't.
                                    pass

                                elif u.get_cargo_space_left() > target_unit.get_cargo_space_left():
                                    # Change targets, because neither target can accept all our resources and
                                    # this target can take more resources.
                                    target_unit = u
                            elif u.type == luxai_constants.Constants.UNIT_TYPES.CART:
                                # Transfer to this cart instead of the current worker target
                                target_unit = u

    # Build the transfer action request
    target_unit_id = None
    if target_unit is not None:
        target_unit_id = target_unit.id

        # Update the transfer amount based on the room of the target
        if target_unit.get_cargo_space_left() < resource_amount:
            resource_amount = target_unit.get_cargo_space_left()

    return luxai_actions.TransferAction(team, unit_id, target_unit_id, resource_type, resource_amount)


class AgentPolicyCHAC(AgentWithModel):
    def __init__(self, mode="train", model=None) -> None:
        """
        Arguments:
            mode: "train" or "inference", which controls if this agent is for training or not.
            model: The pretrained model, or if None it will operate in training mode.
        """
        super().__init__(mode, model)

        self.units_last = 0
        self.city_tiles_last = 0
        self.fuel_collected_last = 0
        self.is_last_turn = False

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.actions_units = [
            partial(luxai_actions.MoveAction, direction=luxai_constants.Constants.DIRECTIONS.CENTER),  # This is the do-nothing action
            partial(luxai_actions.MoveAction, direction=luxai_constants.Constants.DIRECTIONS.NORTH),
            partial(luxai_actions.MoveAction, direction=luxai_constants.Constants.DIRECTIONS.WEST),
            partial(luxai_actions.MoveAction, direction=luxai_constants.Constants.DIRECTIONS.SOUTH),
            partial(luxai_actions.MoveAction, direction=luxai_constants.Constants.DIRECTIONS.EAST),
            partial(smart_transfer_to_nearby, target_type_restriction=luxai_constants.Constants.UNIT_TYPES.CART),  # Transfer to nearby cart
            partial(smart_transfer_to_nearby, target_type_restriction=luxai_constants.Constants.UNIT_TYPES.WORKER),  # Transfer to nearby worker
            luxai_actions.SpawnCityAction,
            luxai_actions.PillageAction,
        ]
        self.actions_cities = [
            luxai_actions.SpawnWorkerAction,
            luxai_actions.SpawnCartAction,
            luxai_actions.ResearchAction,
        ]
        self.action_space = spaces.Discrete(max(len(self.actions_units), len(self.actions_cities)))

        # Observation space: (Basic minimum for a miner agent)
        # Object:
        #   1x is worker
        #   1x is cart
        #   1x is citytile
        #
        #   5x direction_nearest_wood
        #   1x distance_nearest_wood
        #   1x amount
        #
        #   5x direction_nearest_coal
        #   1x distance_nearest_coal
        #   1x amount
        #
        #   5x direction_nearest_uranium
        #   1x distance_nearest_uranium
        #   1x amount
        #
        #   5x direction_nearest_city
        #   1x distance_nearest_city
        #   1x amount of fuel
        #
        #   28x (the same as above, but direction, distance, and amount to the furthest of each)
        #
        #   5x direction_nearest_worker
        #   1x distance_nearest_worker
        #   1x amount of cargo
        # Unit:
        #   1x cargo size
        # State:
        #   1x is night
        #   1x percent of game done
        #   2x citytile counts [cur player, opponent]       62   -11
        #   2x worker counts [cur player, opponent]              -9
        #   2x cart counts [cur player, opponent]                -7
        #   1x research points [cur player]                 67   -5
        #   1x researched coal [cur player]                      -4
        #   1x researched uranium [cur player]                   -3
        #   NEW: 1x fuel quantity of team                   70   -2
        #   NEW: 1x fuel quantity in current unit           71   -1
        self.observation_shape = (3 + 7 * 5 * 2 + 1 + 1 + 1 + 2 + 2 + 2 + 3 + 2,)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float16)

        self.object_nodes = {}

    def get_observation(self, game, unit, city_tile, team, is_new_turn):
        """
        Implements getting a observation from the current game for this unit or city
        """
        observation_index = 0
        if is_new_turn:
            # It's a new turn this event. This flag is set True for only the first observation from each turn.
            # Update any per-turn fixed observation space that doesn't change per unit/city controlled.

            # Build a list of object nodes by type for quick distance-searches
            self.object_nodes = {}

            # Add resources
            for cell in game.map.resources:
                if cell.resource.type not in self.object_nodes:
                    self.object_nodes[cell.resource.type] = np.array([[cell.pos.x, cell.pos.y]])
                else:
                    self.object_nodes[cell.resource.type] = np.concatenate(
                        (
                            self.object_nodes[cell.resource.type],
                            [[cell.pos.x, cell.pos.y]]
                        ),
                        axis=0
                    )

            # Add your own and opponent units
            for t in [team, (team + 1) % 2]:
                for u in game.state["teamStates"][team]["units"].values():
                    key = str(u.type)
                    if t != team:
                        key = str(u.type) + "_opponent"

                    if key not in self.object_nodes:
                        self.object_nodes[key] = np.array([[u.pos.x, u.pos.y]])
                    else:
                        self.object_nodes[key] = np.concatenate(
                            (
                                self.object_nodes[key],
                                [[u.pos.x, u.pos.y]]
                            )
                            , axis=0
                        )

            # Add your own and opponent cities
            for city in game.cities.values():
                for cells in city.city_cells:
                    key = "city"
                    if city.team != team:
                        key = "city_opponent"

                    if key not in self.object_nodes:
                        self.object_nodes[key] = np.array([[cells.pos.x, cells.pos.y]])
                    else:
                        self.object_nodes[key] = np.concatenate(
                            (
                                self.object_nodes[key],
                                [[cells.pos.x, cells.pos.y]]
                            )
                            , axis=0
                        )

        # Observation space: (Basic minimum for a miner agent)
        # Object:
        #   1x is worker
        #   1x is cart
        #   1x is citytile
        #   5x direction_nearest_wood
        #   1x distance_nearest_wood
        #   1x amount
        #
        #   5x direction_nearest_coal
        #   1x distance_nearest_coal
        #   1x amount
        #
        #   5x direction_nearest_uranium
        #   1x distance_nearest_uranium
        #   1x amount
        #
        #   5x direction_nearest_city
        #   1x distance_nearest_city
        #   1x amount of fuel
        #
        #   5x direction_nearest_worker
        #   1x distance_nearest_worker
        #   1x amount of cargo
        #
        #   28x (the same as above, but direction, distance, and amount to the furthest of each)
        #
        # Unit:
        #   1x cargo size
        # State:
        #   1x is night
        #   1x percent of game done
        #   2x citytile counts [cur player, opponent]
        #   2x worker counts [cur player, opponent]
        #   2x cart counts [cur player, opponent]
        #   1x research points [cur player]
        #   1x researched coal [cur player]
        #   1x researched uranium [cur player]
        obs = np.zeros(self.observation_shape)

        # Update the type of this object
        #   1x is worker
        #   1x is cart
        #   1x is citytile
        observation_index = 0
        if unit is not None:
            if unit.type == luxai_constants.Constants.UNIT_TYPES.WORKER:
                obs[observation_index] = 1.0  # Worker
            else:
                obs[observation_index + 1] = 1.0  # Cart
        if city_tile is not None:
            obs[observation_index + 2] = 1.0  # CityTile
        observation_index += 3

        pos = None
        if unit is not None:
            pos = unit.pos
        else:
            pos = city_tile.pos

        if pos is None:
            observation_index += 7 * 5 * 2
        else:
            # Encode the direction to the nearest objects
            #   5x direction_nearest
            #   1x distance
            for distance_function in [closest_node, furthest_node]:
                for key in [
                    luxai_constants.Constants.RESOURCE_TYPES.WOOD,
                    luxai_constants.Constants.RESOURCE_TYPES.COAL,
                    luxai_constants.Constants.RESOURCE_TYPES.URANIUM,
                    "city",
                    str(luxai_constants.Constants.UNIT_TYPES.WORKER)]:
                    # Process the direction to and distance to this object type

                    # Encode the direction to the nearest object (excluding itself)
                    #   5x direction
                    #   1x distance
                    if key in self.object_nodes:
                        if (
                                (key == "city" and city_tile is not None) or
                                (unit is not None and str(unit.type) == key and len(
                                    game.map.get_cell_by_pos(unit.pos).units) <= 1)
                        ):
                            # Filter out the current unit from the closest-search
                            closest_index = closest_node((pos.x, pos.y), self.object_nodes[key])
                            filtered_nodes = np.delete(self.object_nodes[key], closest_index, axis=0)
                        else:
                            filtered_nodes = self.object_nodes[key]

                        if len(filtered_nodes) == 0:
                            # No other object of this type
                            obs[observation_index + 5] = 1.0
                        else:
                            # There is another object of this type
                            closest_index = distance_function((pos.x, pos.y), filtered_nodes)

                            if closest_index is not None and closest_index >= 0:
                                closest = filtered_nodes[closest_index]
                                closest_position = Position(closest[0], closest[1])
                                direction = pos.direction_to(closest_position)
                                mapping = {
                                    luxai_constants.Constants.DIRECTIONS.CENTER: 0,
                                    luxai_constants.Constants.DIRECTIONS.NORTH: 1,
                                    luxai_constants.Constants.DIRECTIONS.WEST: 2,
                                    luxai_constants.Constants.DIRECTIONS.SOUTH: 3,
                                    luxai_constants.Constants.DIRECTIONS.EAST: 4,
                                }
                                obs[observation_index + mapping[direction]] = 1.0  # One-hot encoding direction

                                # 0 to 1 distance
                                distance = pos.distance_to(closest_position)
                                obs[observation_index + 5] = min(distance / 20.0, 1.0)

                                # 0 to 1 value (amount of resource, cargo for unit, or fuel for city)
                                if key == "city":
                                    # City fuel as % of upkeep for 200 turns
                                    c = game.cities[game.map.get_cell_by_pos(closest_position).city_tile.city_id]
                                    obs[observation_index + 6] = min(
                                        c.fuel / (c.get_light_upkeep() * 200.0),
                                        1.0
                                    )
                                elif key in [luxai_constants.Constants.RESOURCE_TYPES.WOOD,
                                             luxai_constants.Constants.RESOURCE_TYPES.COAL,
                                             luxai_constants.Constants.RESOURCE_TYPES.URANIUM]:
                                    # Resource amount
                                    obs[observation_index + 6] = min(
                                        game.map.get_cell_by_pos(closest_position).resource.amount / 500,
                                        1.0
                                    )
                                else:
                                    # Unit cargo
                                    obs[observation_index + 6] = min(
                                        next(iter(game.map.get_cell_by_pos(
                                            closest_position).units.values())).get_cargo_space_left() / 100,
                                        1.0
                                    )

                    observation_index += 7

        if unit is not None:
            # Encode the cargo space
            #   1x cargo size
            obs[observation_index] = unit.get_cargo_space_left() / GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"][
                "WORKER"]
            observation_index += 1
        else:
            observation_index += 1

        # Game state observations

        #   1x is night
        obs[observation_index] = game.is_night()
        observation_index += 1

        #   1x percent of game done
        obs[observation_index] = game.state["turn"] / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]
        observation_index += 1

        #   2x citytile counts [cur player, opponent]
        #   2x worker counts [cur player, opponent]
        #   2x cart counts [cur player, opponent]
        max_count = 30
        for key in ["city", str(luxai_constants.Constants.UNIT_TYPES.WORKER), str(luxai_constants.Constants.UNIT_TYPES.CART)]:
            if key in self.object_nodes:
                obs[observation_index] = len(self.object_nodes[key]) / max_count
            if (key + "_opponent") in self.object_nodes:
                obs[observation_index + 1] = len(self.object_nodes[(key + "_opponent")]) / max_count
            observation_index += 2

        #   1x research points [cur player]
        #   1x researched coal [cur player]
        #   1x researched uranium [cur player]
        obs[observation_index] = game.state["teamStates"][team]["researchPoints"] / 200.0
        obs[observation_index + 1] = float(game.state["teamStates"][team]["researched"]["coal"])
        obs[observation_index + 2] = float(game.state["teamStates"][team]["researched"]["uranium"])
        observation_index += 3

        obs[observation_index] = game.state["teamStats"][self.team]["fuelGenerated"]
        if unit is not None:
            obs[observation_index + 1] = unit.get_cargo_fuel_value()
        elif city_tile is not None:
            obs[observation_index + 1] = game.cities[city_tile.city_id].fuel
        else:
            obs[observation_index + 1] = 0.0

        return obs

    def action_code_to_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            action_code: Index of action to take into the action array.
        Returns: An action.
        """
        # Map action_code index into to a constructed Action object
        try:
            x = None
            y = None
            if city_tile is not None:
                x = city_tile.pos.x
                y = city_tile.pos.y
            elif unit is not None:
                x = unit.pos.x
                y = unit.pos.y

            if city_tile is not None:
                action = self.actions_cities[action_code % len(self.actions_cities)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y
                )
            else:
                action = self.actions_units[action_code % len(self.actions_units)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y
                )

            return action
        except Exception as e:
            # Not a valid action
            print(e)
            return None

    def take_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            actionCode: Index of action to take into the action array.
        """
        action = self.action_code_to_action(action_code, game, unit, city_tile, team)
        self.match_controller.take_action(action)

    def game_start(self, game):
        """
        This function is called at the start of each game. Use this to
        reset and initialize per game. Note that self.team may have
        been changed since last game. The game map has been created
        and starting units placed.

        Args:
            game ([type]): Game.
        """
        self.units_last = 0
        self.city_tiles_last = 0
        self.fuel_collected_last = 0

        # TODO: reset/init model

    def get_reward(self, game, is_game_finished, is_new_turn, is_game_error):
        """
        Returns the reward function for this step of the game. Reward should be a
        delta increment to the reward, not the total current reward.
        """
        # TODO: Reward shaping?
        # Fuel value decrease with time, incentivize getting fuel early?

        if is_game_error:
            # Game environment step failed, assign a game lost reward to not incentivise this
            print("Game failed due to error")
            return -1.0

        if not is_new_turn and not is_game_finished:
            # Only apply rewards at the start of each turn or at game end
            return 0

        # Get some basic stats
        unit_count = len(game.state["teamStates"][self.team]["units"])

        city_count = 0
        city_count_opponent = 0
        city_tile_count = 0
        city_tile_count_opponent = 0
        for city in game.cities.values():
            if city.team == self.team:
                city_count += 1
            else:
                city_count_opponent += 1

            for cell in city.city_cells:
                if city.team == self.team:
                    city_tile_count += 1
                else:
                    city_tile_count_opponent += 1

        rewards = {}

        # Give a reward for unit creation/death. 0.05 reward per unit.
        rewards["rew/r_units"] = (unit_count - self.units_last) * 0.05
        self.units_last = unit_count

        # Give a reward for city creation/death. 0.1 reward per city.
        rewards["rew/r_city_tiles"] = (city_tile_count - self.city_tiles_last) * 0.1
        self.city_tiles_last = city_tile_count

        # Reward collecting fuel
        fuel_collected = game.stats["teamStats"][self.team]["fuelGenerated"]
        rewards["rew/r_fuel_collected"] = ((fuel_collected - self.fuel_collected_last) / 20000)
        self.fuel_collected_last = fuel_collected

        # Give a reward of 1.0 per city tile alive at the end of the game
        rewards["rew/r_city_tiles_end"] = 0
        if is_game_finished:
            self.is_last_turn = True
            rewards["rew/r_city_tiles_end"] = city_tile_count

            # Example of a game win/loss reward instead
            if game.get_winning_team() == self.team:
                rewards["rew/r_game_win"] = 10.0  # Win
            else:
                rewards["rew/r_game_win"] = -10.0  # Loss

        reward = 0
        for name, value in rewards.items():
            reward += value

        return reward

    def process_turn(self, game, team):
        """
        Decides on a set of actions for the current turn. Not used in training, only inference. Generally
        don't modify this part of the code.
        Returns: Array of actions to perform.
        """
        start_time = time.time()
        actions = []
        new_turn = True

        # Inference the model per-unit
        units = game.state["teamStates"][team]["units"].values()
        for unit in units:
            if unit.can_act():
                obs = self.get_observation(game, unit, None, unit.team, new_turn)
                action_code, _states = self.model.predict(obs, unit.id)
                if action_code is not None:
                    actions.append(
                        self.action_code_to_action(action_code, game=game, unit=unit, city_tile=None, team=unit.team))
                new_turn = False

        # Inference the model per-city
        cities = game.cities.values()
        for city in cities:
            if city.team == team:
                for cell in city.city_cells:
                    city_tile = cell.city_tile
                    if city_tile.can_act():
                        obs = self.get_observation(game, None, city_tile, city.team, new_turn)
                        action_code, _states = self.model.predict(obs, city.id)
                        if action_code is not None:
                            actions.append(
                                self.action_code_to_action(action_code, game=game, unit=None, city_tile=city_tile,
                                                           team=city.team))
                        new_turn = False

        time_taken = time.time() - start_time
        if time_taken > 0.5:  # Warn if larger than 0.5 seconds.
            print("WARNING: Inference took %.3f seconds for computing actions. Limit is 1 second." % time_taken,
                  file=sys.stderr)

        return actions
