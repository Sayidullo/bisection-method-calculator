from flask import Flask, render_template, request
import sympy as sp
import numpy as np

# Import additional parsing tools
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)

app = Flask(__name__)

# Allowed mathematical functions and constants
allowed_locals = {
    'ln': sp.log,
    'log': sp.log,
    'log10': lambda arg: sp.log(arg, 10),
    'sin': sp.sin,
    'cos': sp.cos,
    'tan': sp.tan,
    'sec': lambda x: 1/sp.cos(x),
    'cosec': lambda x: 1/sp.sin(x),
    'cot': lambda x: 1/sp.tan(x),
    'sqrt': sp.sqrt,
    'exp': sp.exp,
    'abs': sp.Abs,
    'pi': sp.pi,
    'E': sp.E
}

transformations = standard_transformations + (implicit_multiplication_application, convert_xor)

def bisection_method(f, a, b, epsilon=0.001, max_iterations=100):
    """Performs the Bisection method to find a root of f(x)."""
    steps = []
    
    try:
        f_a = f(a)
        f_b = f(b)

        if not np.isfinite(f_a) or not np.isfinite(f_b):
            return None, "Function evaluation resulted in an invalid number."

        if f_a * f_b > 0:
            return None, "Invalid interval: f(a) and f(b) must have opposite signs."
    
    except Exception as e:
        return None, f"Function evaluation error: {str(e)}"

    for iteration in range(max_iterations):
        mid = (a + b) / 2
        f_mid = f(mid)
        error_val = abs(b - a) / 2

        steps.append({
            "iteration": iteration + 1,
            "a": a,
            "b": b,
            "mid": mid,
            "f_mid": f_mid,
            "error": error_val,
            "formula": f"mid = ({a:.6f} + {b:.6f}) / 2 = {mid:.6f}"
        })

        if abs(f_mid) < epsilon or error_val < epsilon:
            return steps, None

        if f_mid * f_a < 0:
            b = mid
            f_b = f_mid  # Update f(b)
        else:
            a = mid
            f_a = f_mid  # Update f(a)

    return None, "Bisection method did not converge after 100 iterations."

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        function_str = request.form.get("function", "").strip()
        
        function_str = function_str.replace("−", "-")  # Fix Unicode minus sign
        function_str = function_str.replace("X", "x")  # Convert 'X' to 'x' for consistency
        function_str = function_str.lower()  # Convert the entire input to lowercase

        if not function_str:
            return render_template("index.html", error="Function input cannot be empty.")

        try:
            x = sp.symbols('x')
            f_expr = parse_expr(function_str, local_dict=allowed_locals, transformations=transformations)
        except (SyntaxError, ValueError, TypeError) as e:
            return render_template("index.html", error=f"Invalid function expression: {str(e)}")

        try:
            f = sp.lambdify(x, f_expr, 'numpy')  # Try NumPy first
        except:
            f = sp.lambdify(x, f_expr, 'sympy')  # Fallback to SymPy

        try:
            a = float(request.form.get("a"))
            b = float(request.form.get("b"))
            epsilon = float(request.form.get("epsilon")) if request.form.get("epsilon") else 0.001

            if a >= b:
                return render_template("index.html", error="Invalid interval: a must be less than b.")
            if epsilon < 1e-10:
                return render_template("index.html", error="Epsilon is too small. Use a value ≥ 1e-10.")

        except ValueError:
            return render_template("index.html", error="Invalid numerical input for a, b, or epsilon.")

        steps, error = bisection_method(f, a, b, epsilon)

        if error:
            return render_template("index.html", error=error)

        solution = steps[-1]["mid"]
        iteration_points = [step["mid"] for step in steps]
        iteration_y = [f(val) for val in iteration_points]

        min_x = min(iteration_points)
        max_x = max(iteration_points)
        margin = (max_x - min_x) * 0.5 if max_x != min_x else 1
        plot_min = min_x - margin
        plot_max = max_x + margin

        x_vals = np.linspace(plot_min, plot_max, 200)
        y_vals = np.array([f(x) if np.isfinite(f(x)) else np.nan for x in x_vals])  # Prevent overflow

        function_latex = sp.latex(f_expr)

        return render_template(
            "result.html",
            steps=steps,
            solution=solution,
            iterations=len(steps),
            x_vals=list(x_vals),
            y_vals=list(y_vals),
            iteration_points=iteration_points,
            iteration_y=iteration_y,
            function_str=function_latex
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
