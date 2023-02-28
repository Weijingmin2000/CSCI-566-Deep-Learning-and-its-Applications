rm -f assignment1.zip 
zip -r assignment1.zip . -i "*.ipynb" "lib/*.py" "lib/mlp/*.py" "lib/cnn/*.py" "problem_1_solution.pdf" "problem_2_solution.pdf" -x "*.ipynb_checkpoints*"
