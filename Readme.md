# Deep Learning for g-Expectation Problems

This project uses deep learning techniques to solve g-expectation problems in finance, focusing on different stochastic models and generator functions. G-expectations are a type of nonlinear expectation that generalizes classical linear expectations, providing a framework for modeling uncertainty and risk in financial markets.

## Overview

The code implements a deep learning approach to solve partial differential equations (PDEs) arising from nonlinear expectation problems. It supports three different stochastic models:

1. Constant volatility model
2. Black-Scholes-Merton (BSM) model
3. Ornstein-Uhlenbeck Mean-Reverting (OUMR) model

And four different generator functions:

1. $g_0(t, y, z) = 0$
2. $g_1(t, y, z) = 2 \sqrt{z^2 + \epsilon}$
3. $g_2(t, y, z) = y + \sqrt{z^2 + \epsilon}$
4. $g_3(t, y, z) = (1/e) \exp(y) + \sqrt{z^2 + \epsilon}$

## Branch Information

- **Main Branch**: The main branch contains the code used to generate all the data for the final MSc thesis submission.
- **Post-submission-edits**: A new branch has been created that successfully trains the model with only the boundary condition v(t,x) = -x.

## Requirements

- Python 3.x
- DeepXDE
- TensorFlow
- NumPy
- Matplotlib

## Usage

1. Set the `model_type` and `generator_name` variables in the script to choose the desired model and generator function.
2. Run the script:

   ```
   python main.py
   ```

3. The script will train the model and generate plots of the results.

## Output

The script generates the following outputs:

- A contour plot of the solution
- A plot of the training loss history
- Saved model checkpoints

These outputs are saved in a directory named after the chosen model type.

## License

This project is licensed under the GNU Lesser General Public License v2.1 (LGPL-2.1).

