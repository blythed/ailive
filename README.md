## Usage

To start the visual generation:

1. Open "terminal" (black thing on the sidebar)
1. Type `cd ~/Deeplearning/ailive` - this navigates to the project
1. Type `python3.6 app.py`
1. If this doesn't work type: `ps ax|grep python|cut -c1-5|xargs kill -9`
1. Then retype `python3.6 app.py`
1. Navigate in the browser to `localhost:5000` to view the visuals
1. press `F11` for full screen
1. To get the control pad navigate to `localhost:5000/controls` into the browser
1. To quit select the terminal again and press `ctrl+c` or close the window

## Notes

1. Sensitivity tensor controls input to the forward pass
1. Dimensions `0:L` control response to music `L:LE` control.
1. Slider for `u/i`
1. Slider for `t/z`
1. Slider for size of input noise `Wpx`
