# Simple Race Car Optimization Using Genetic Algorithm

## Objective

This was an energy system optimization competition. We were given energy system constraints and data. The goal was to build a model of the race car and create an optimization model to operate the car. The car that makes the best use of the available resources and finishes the race in the shortest time wins.

This model achieved 3rd place! 

## Optimization Strategy:

My strategy was to optimize for a single lap while taking into consideration the total number of laps in the race. The Genetic Algorithim takes on an elitist strategy to minimize the total time. The objective function is written in terms of the variable velocity.