"""
geoplot.py
----------

This visualization renders a 3-D plot of the data given the state
trajectory of a simulation, and the path of the property to render.

It generates an HTML file that contains code to render the plot
using Cesium Ion, and the GeoJSON file of data provided to the plot.

An example of its usage is as follows:

    ```python
    from agent_torch.visualize import GeoPlot

# create a simulation
# ...

# create a visualizer
engine = GeoPlot(config, {
  cesium_token: "...",
  step_time: 3600,
  coordinates = "agents/consumers/coordinates",
  feature = "agents/consumers/money_spent",
})

# visualize in the runner-loop
for i in range(0, num_episodes):
  runner.step(num_steps_per_episode)
  engine.render(runner.state_trajectory)
```
"""

import re
import json

import pandas as pd
import numpy as np

from string import Template  # For HTML template substitution
from agent_torch.core.helpers import get_by_path  # For nested dict navigation

# HTML template for the Cesium-based 3D visualization viewer
# This template contains embedded JavaScript for rendering time-series geospatial data
# Key JavaScript functions included:
# - interpolateColor(): Creates color gradients for value mapping
# - getColor(): Maps data values to blue-red color scale
# - getPixelSize(): Maps data values to point sizes for size-based visualization
# - processTimeSeriesData(): Parses GeoJSON and extracts time-series information
# - createTimeSeriesEntities(): Creates Cesium entities with time-varying properties
# Template variables: $accessToken, $startTime, $stopTime, $data, $visualType
geoplot_template = """
<!doctype html>
<html lang="en">
	<head>
		<meta charset="UTF-8" />
		<meta
			name="viewport"
			content="width=device-width, initial-scale=1.0"
		/>
		<title>Cesium Time-Series Heatmap Visualization</title>
		<script src="https://cesium.com/downloads/cesiumjs/releases/1.95/Build/Cesium/Cesium.js"></script>
		<link
			href="https://cesium.com/downloads/cesiumjs/releases/1.95/Build/Cesium/Widgets/widgets.css"
			rel="stylesheet"
		/>
		<style>
			#cesiumContainer {
				width: 100%;
				height: 100%;
			}
		</style>
	</head>
	<body>
		<div id="cesiumContainer"></div>
		<script>
			// Your Cesium ion access token here
			Cesium.Ion.defaultAccessToken = '$accessToken'

			// Create the viewer
			const viewer = new Cesium.Viewer('cesiumContainer')

			function interpolateColor(color1, color2, factor) {
				const result = new Cesium.Color()
				result.red = color1.red + factor * (color2.red - color1.red)
				result.green =
					color1.green + factor * (color2.green - color1.green)
				result.blue = color1.blue + factor * (color2.blue - color1.blue)
				result.alpha = '$visualType' == 'size' ? 0.2 :
					color1.alpha + factor * (color2.alpha - color1.alpha)
				return result
			}

			function getColor(value, min, max) {
				const factor = (value - min) / (max - min)
				return interpolateColor(
					Cesium.Color.BLUE,
					Cesium.Color.RED,
					factor
				)
			}

			function getPixelSize(value, min, max) {
				const factor = (value - min) / (max - min)
				return 100 * (1 + factor)
			}

			function processTimeSeriesData(geoJsonData) {
				const timeSeriesMap = new Map()
				let minValue = Infinity
				let maxValue = -Infinity

				geoJsonData.features.forEach((feature) => {
					const id = feature.properties.id
					const time = Cesium.JulianDate.fromIso8601(
						feature.properties.time
					)
					const value = feature.properties.value
					const coordinates = feature.geometry.coordinates

					if (!timeSeriesMap.has(id)) {
						timeSeriesMap.set(id, [])
					}
					timeSeriesMap.get(id).push({ time, value, coordinates })

					minValue = Math.min(minValue, value)
					maxValue = Math.max(maxValue, value)
				})

				return { timeSeriesMap, minValue, maxValue }
			}

			function createTimeSeriesEntities(
				timeSeriesData,
				startTime,
				stopTime
			) {
				const dataSource = new Cesium.CustomDataSource(
					'AgentTorch Simulation'
				)

				for (const [id, timeSeries] of timeSeriesData.timeSeriesMap) {
					const entity = new Cesium.Entity({
						id: id,
						availability: new Cesium.TimeIntervalCollection([
							new Cesium.TimeInterval({
								start: startTime,
								stop: stopTime,
							}),
						]),
						position: new Cesium.SampledPositionProperty(),
						point: {
							pixelSize: '$visualType' == 'size' ? new Cesium.SampledProperty(Number) : 10,
							color: new Cesium.SampledProperty(Cesium.Color),
						},
						properties: {
							value: new Cesium.SampledProperty(Number),
						},
					})

					timeSeries.forEach(({ time, value, coordinates }) => {
						const position = Cesium.Cartesian3.fromDegrees(
							coordinates[0],
							coordinates[1]
						)
						entity.position.addSample(time, position)
						entity.properties.value.addSample(time, value)
						entity.point.color.addSample(
							time,
							getColor(
								value,
								timeSeriesData.minValue,
								timeSeriesData.maxValue
							)
						)

						if ('$visualType' == 'size') {
						  entity.point.pixelSize.addSample(
  							time,
  							getPixelSize(
  								value,
  								timeSeriesData.minValue,
  								timeSeriesData.maxValue
  							)
  						)
						}
					})

					dataSource.entities.add(entity)
				}

				return dataSource
			}

			// Example time-series GeoJSON data
			const geoJsons = $data

			const start = Cesium.JulianDate.fromIso8601('$startTime')
			const stop = Cesium.JulianDate.fromIso8601('$stopTime')

			viewer.clock.startTime = start.clone()
			viewer.clock.stopTime = stop.clone()
			viewer.clock.currentTime = start.clone()
			viewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP
			viewer.clock.multiplier = 3600 // 1 hour per second

			viewer.timeline.zoomTo(start, stop)

			for (const geoJsonData of geoJsons) {
				const timeSeriesData = processTimeSeriesData(geoJsonData)
				const dataSource = createTimeSeriesEntities(
					timeSeriesData,
					start,
					stop
				)
				viewer.dataSources.add(dataSource)
				viewer.zoomTo(dataSource)
			}
		</script>
	</body>
</html>
"""


def read_var(state, var):
    """Extract a variable from nested simulation state using a path string.
    
    This utility function navigates through nested dictionaries using a
    slash-separated path string to retrieve values from the simulation state.
    
    Args:
        state (dict): The nested simulation state dictionary.
        var (str): A slash-separated path string (e.g., "agents/consumers/coordinates").
                  Each part represents a key in the nested dictionary structure.
    
    Returns:
        Any: The value found at the specified path in the state dictionary.
        
    Example:
        >>> state = {"agents": {"consumers": {"money": [100, 200, 300]}}}
        >>> read_var(state, "agents/consumers/money")
        [100, 200, 300]
    """
    return get_by_path(state, re.split("/", var))


class GeoPlot:
    """Interactive 3D geographic visualization for AgentTorch simulations.
    
    This class creates interactive 3D visualizations of agent-based simulation
    data using Cesium Ion technology. It processes simulation state trajectories
    and generates HTML files with embedded Cesium viewers that display agent
    properties as time-series data on a 3D globe.
    
    The visualization supports two main display modes:
    - Color-based: Agent properties mapped to color intensity (blue=low, red=high)
    - Size-based: Agent properties mapped to point size (small=low, large=high)
    
    Attributes:
        config (dict): Simulation configuration containing metadata.
        cesium_token (str): Cesium Ion access token for 3D rendering.
        step_time (int): Time interval between simulation steps in seconds.
        entity_position (str): Path to agent coordinate data in simulation state.
        entity_property (str): Path to agent property data to visualize.
        visualization_type (str): Visualization mode ('color' or 'size').
    """
    
    def __init__(self, config, options):
        """Initialize the GeoPlot visualization engine.
        
        Args:
            config (dict): Simulation configuration dictionary containing:
                - simulation_metadata (dict): Metadata with simulation name,
                  number of episodes, and steps per episode.
            options (dict): Visualization options containing:
                - cesium_token (str): Cesium Ion access token for 3D rendering.
                - step_time (int): Time interval between steps in seconds.
                - coordinates (str): Path to agent coordinates in state
                  (e.g., "agents/consumers/coordinates").
                - feature (str): Path to property to visualize
                  (e.g., "agents/consumers/money_spent").
                - visualization_type (str): Either 'color' or 'size'.
                
        Raises:
            KeyError: If required options are missing from the options dict.
            
        Example:
            >>> config = {
            ...     "simulation_metadata": {
            ...         "name": "my_simulation",
            ...         "num_episodes": 1,
            ...         "num_steps_per_episode": 24
            ...     }
            ... }
            >>> options = {
            ...     "cesium_token": "your_token_here",
            ...     "step_time": 3600,
            ...     "coordinates": "agents/consumers/coordinates",
            ...     "feature": "agents/consumers/money_spent",
            ...     "visualization_type": "color"
            ... }
            >>> geoplot = GeoPlot(config, options)
        """
        self.config = config
        
        # Extract and store visualization configuration options
        # Using tuple unpacking for efficient assignment of multiple attributes
        (
            self.cesium_token,
            self.step_time,
            self.entity_position,
            self.entity_property,
            self.visualization_type,
        ) = (
            options["cesium_token"],
            options["step_time"],
            options["coordinates"],
            options["feature"],
            options["visualization_type"],
        )

    def render(self, state_trajectory):
        """Generate 3D visualization files from simulation trajectory.
        
		Creates GeoJSON data file and HTML viewer with Cesium 3D visualization.
        
        The method extracts agent coordinates and property values from each simulation
        step, creates timestamps based on the step_time interval, and generates
        GeoJSON features for each agent at each time step. The resulting data is
        then embedded into a Cesium-based HTML visualization.
        
        Args:
            state_trajectory (list): A list of simulation episodes, where each episode
                contains a list of simulation steps. Each step is a nested dictionary
                containing the simulation state at that point in time.
                Structure: [episode1, episode2, ...] where each episode is
                [step1, step2, ...] and each step contains agent data.
                
        Raises:
            KeyError: If the specified entity_position or entity_property paths
                are not found in the simulation state.
            IOError: If the output files cannot be written to disk.
            
        Side Effects:
            Creates two files in the current directory:
            - {simulation_name}.geojson: Contains the time-series geospatial data
            - {simulation_name}.html: Contains the interactive Cesium visualization
            
        Example:
            >>> # After running a simulation with state_trajectory
            >>> geoplot.render(runner.state_trajectory)
            # Creates 'my_simulation.geojson' and 'my_simulation.html'
        """
        # Initialize containers for coordinates and property values
        coords, values = [], []
        
        # Extract simulation name for output file naming
        name = self.config["simulation_metadata"]["name"]
        geodata_path, geoplot_path = f"{name}.geojson", f"{name}.html"

        # Process each episode in the state trajectory
        # Note: We iterate to len-1 to avoid potential index errors with final_state access
        for i in range(0, len(state_trajectory) - 1):
            # Get the final state of each episode for data extraction
            final_state = state_trajectory[i][-1]

            # Extract agent coordinates and convert to standard list format
            # Coordinates are expected to be in [lat, lon] format
            coords = np.array(read_var(final_state, self.entity_position)).tolist()
            
            # Extract agent property values, flatten any nested arrays, and store
            # This handles cases where properties might be multi-dimensional
            values.append(
                np.array(read_var(final_state, self.entity_property)).flatten().tolist()
            )

        # Generate timestamps for the time-series visualization
        # Uses current UTC time as the starting point for the visualization
        start_time = pd.Timestamp.utcnow()
        timestamps = [
            start_time + pd.Timedelta(seconds=i * self.step_time)
            for i in range(
                self.config["simulation_metadata"]["num_episodes"]
                * self.config["simulation_metadata"]["num_steps_per_episode"]
            )
        ]

        # Create GeoJSON feature collections for each agent
        geojsons = []
        for i, coord in enumerate(coords):
            features = []
            # Create a feature for each time step with agent position and property value
            for time, value_list in zip(timestamps, values):
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
							# Note: GeoJSON uses [longitude, latitude] format, so we swap coordinates
							# coord contains [lat, lon] format
                            "coordinates": [coord[1], coord[0]],
                        },
                        "properties": {
                            "value": value_list[i],  # Agent's property value at this time
                            "time": time.isoformat(),  # ISO 8601 formatted timestamp
                        },
                    }
                )
            # Group all features for this agent into a FeatureCollection
            geojsons.append({"type": "FeatureCollection", "features": features})

        # Write the GeoJSON data to file with proper UTF-8 encoding
        with open(geodata_path, "w", encoding="utf-8") as f:
            json.dump(geojsons, f, ensure_ascii=False, indent=2)

        # Generate the HTML visualization file using the template
        tmpl = Template(geoplot_template)
        with open(geoplot_path, "w", encoding="utf-8") as f:
            # Substitute template variables with actual configuration values
            f.write(
                tmpl.substitute(
                    {
                        "accessToken": self.cesium_token,  # Cesium Ion access token
                        "startTime": timestamps[0].isoformat(),  # Simulation start time
                        "stopTime": timestamps[-1].isoformat(),  # Simulation end time
                        "data": json.dumps(geojsons),  # Embedded GeoJSON data
                        "visualType": self.visualization_type,  # Color or size visualization
                    }
                )
            )
