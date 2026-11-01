import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import os
import pickle
import sys
import types
import pandas as pd
import numpy as np
from flask import Flask, request, render_template_string

# --- COMPATIBILITY PATCH FOR MODEL UNPICKLING ---
try:
    import sklearn._loss
    if '_loss' not in sys.modules:
        loss_proxy = types.ModuleType('_loss')
        for attr in dir(sklearn._loss):
            setattr(loss_proxy, attr, getattr(sklearn._loss, attr))
        if not hasattr(loss_proxy, 'CyHalfSquaredError'):
            class CyHalfSquaredError: pass
            loss_proxy.CyHalfSquaredError = CyHalfSquaredError
        if not hasattr(loss_proxy, 'HalfSquaredError'):
            class HalfSquaredError: pass
            loss_proxy.HalfSquaredError = HalfSquaredError
        sys.modules['_loss'] = loss_proxy
except Exception:
    pass
# -----------------------------------------------

app = Flask(__name__)

# Locate and load the Gradient Boosting model
MODEL_PATHS = ['GradientBoost.pkl']
model = None

for path in MODEL_PATHS:
    if os.path.exists(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"Successfully loaded model from {path}")
        break

FEATURE_NAMES = [
    'Name',
    'Department',
    'Experience_Years',
    'Education_Level',
    'Age',
    'Gender',
    'City'
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gradient Boosting Analytics Suite</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>body { font-family: 'Inter', sans-serif; }</style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex items-center justify-center p-4">
    <div class="w-full max-w-4xl bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl overflow-hidden">
        
        <!-- Header Banner -->
        <div class="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 p-6 text-center">
            <h1 class="text-2xl font-bold tracking-tight text-white">Employee Compensation Intelligence</h1>
            <p class="text-blue-100 text-sm mt-1">Powered by Scikit-Learn Gradient Boosting Regressor</p>
        </div>

        <!-- Navigation Tabs -->
        <div class="flex border-b border-slate-800 bg-slate-900/50 px-6 pt-4 gap-4">
            <button onclick="switchTab('single')" id="btn-single" class="pb-3 text-sm font-semibold border-b-2 transition-all {% if active_tab != 'batch' and active_tab != 'analytics' %}border-blue-500 text-blue-400{% else %}border-transparent text-slate-400 hover:text-slate-200{% endif %}">Single Prediction</button>
            <button onclick="switchTab('batch')" id="btn-batch" class="pb-3 text-sm font-semibold border-b-2 transition-all {% if active_tab == 'batch' %}border-blue-500 text-blue-400{% else %}border-transparent text-slate-400 hover:text-slate-200{% endif %}">Batch Simulator</button>
            <button onclick="switchTab('analytics')" id="btn-analytics" class="pb-3 text-sm font-semibold border-b-2 transition-all {% if active_tab == 'analytics' %}border-blue-500 text-blue-400{% else %}border-transparent text-slate-400 hover:text-slate-200{% endif %}">Model Insights</button>
        </div>

        <div class="p-8">
            <!-- TAB 1: SINGLE PREDICTION -->
            <div id="tab-single" class="tab-pane {% if active_tab == 'batch' or active_tab == 'analytics' %}hidden{% endif %}">
                <form method="POST" class="space-y-6">
                    <input type="hidden" name="action_type" value="single">
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div class="md:col-span-2">
                            <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Employee Full Name</label>
                            <input type="text" name="Name" required value="{{ form_data.Name if form_data and active_tab=='single' else '' }}" placeholder="e.g. Sarah Jenkins" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-blue-500 transition-colors">
                        </div>

                        <div>
                            <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Department</label>
                            <select name="Department" required class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-blue-500 transition-colors">
                                <option value="" disabled {% if not form_data %}selected{% endif %}>Select Department</option>
                                <option value="Engineering" {% if form_data and form_data.Department == 'Engineering' %}selected{% endif %}>Engineering</option>
                                <option value="Data Science" {% if form_data and form_data.Department == 'Data Science' %}selected{% endif %}>Data Science</option>
                                <option value="Sales" {% if form_data and form_data.Department == 'Sales' %}selected{% endif %}>Sales</option>
                                <option value="Marketing" {% if form_data and form_data.Department == 'Marketing' %}selected{% endif %}>Marketing</option>
                                <option value="Finance" {% if form_data and form_data.Department == 'Finance' %}selected{% endif %}>Finance</option>
                                <option value="Human Resources" {% if form_data and form_data.Department == 'Human Resources' %}selected{% endif %}>Human Resources</option>
                            </select>
                        </div>

                        <div>
                            <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Education Level</label>
                            <select name="Education_Level" required class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-blue-500 transition-colors">
                                <option value="" disabled {% if not form_data %}selected{% endif %}>Select Level</option>
                                <option value="Bachelor's" {% if form_data and form_data.Education_Level == "Bachelor's" %}selected{% endif %}>Bachelor's</option>
                                <option value="Master's" {% if form_data and form_data.Education_Level == "Master's" %}selected{% endif %}>Master's</option>
                                <option value="PhD" {% if form_data and form_data.Education_Level == 'PhD' %}selected{% endif %}>PhD</option>
                                <option value="High School" {% if form_data and form_data.Education_Level == 'High School' %}selected{% endif %}>High School</option>
                            </select>
                        </div>

                        <div>
                            <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Experience (Years)</label>
                            <input type="number" step="0.5" min="0" max="50" name="Experience_Years" required value="{{ form_data.Experience_Years if form_data and active_tab=='single' else '' }}" placeholder="e.g. 6.5" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-blue-500 transition-colors">
                        </div>

                        <div>
                            <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Age</label>
                            <input type="number" min="18" max="80" name="Age" required value="{{ form_data.Age if form_data and active_tab=='single' else '' }}" placeholder="e.g. 32" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-blue-500 transition-colors">
                        </div>

                        <div>
                            <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Gender</label>
                            <select name="Gender" required class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-blue-500 transition-colors">
                                <option value="" disabled {% if not form_data %}selected{% endif %}>Select Gender</option>
                                <option value="Female" {% if form_data and form_data.Gender == 'Female' %}selected{% endif %}>Female</option>
                                <option value="Male" {% if form_data and form_data.Gender == 'Male' %}selected{% endif %}>Male</option>
                                <option value="Non-Binary" {% if form_data and form_data.Gender == 'Non-Binary' %}selected{% endif %}>Non-Binary</option>
                            </select>
                        </div>

                        <div>
                            <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">City Location</label>
                            <select name="City" required class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-blue-500 transition-colors">
                                <option value="" disabled {% if not form_data %}selected{% endif %}>Select City</option>
                                <option value="New York" {% if form_data and form_data.City == 'New York' %}selected{% endif %}>New York</option>
                                <option value="San Francisco" {% if form_data and form_data.City == 'San Francisco' %}selected{% endif %}>San Francisco</option>
                                <option value="London" {% if form_data and form_data.City == 'London' %}selected{% endif %}>London</option>
                                <option value="Austin" {% if form_data and form_data.City == 'Austin' %}selected{% endif %}>Austin</option>
                                <option value="Seattle" {% if form_data and form_data.City == 'Seattle' %}selected{% endif %}>Seattle</option>
                                <option value="Remote" {% if form_data and form_data.City == 'Remote' %}selected{% endif %}>Remote</option>
                            </select>
                        </div>
                    </div>

                    <button type="submit" class="w-full bg-blue-600 hover:bg-blue-500 text-white font-semibold py-3 px-6 rounded-lg transition-colors shadow-lg shadow-blue-600/20">Generate Compensation Estimate</button>
                </form>

                {% if prediction and active_tab == 'single' %}
                <div class="mt-8 p-6 bg-slate-800/80 border border-emerald-500/30 rounded-xl text-center relative overflow-hidden">
                    <div class="absolute top-0 left-0 w-1.5 h-full bg-emerald-500"></div>
                    <h3 class="text-xs font-bold uppercase tracking-wider text-emerald-400 mb-1">Estimated Projection for {{ employee_name }}</h3>
                    <div class="text-3xl font-extrabold text-white mt-2">${{ prediction }}</div>
                    <p class="text-xs text-slate-400 mt-2">Ensemble Stage Spread (Confidence Variance): ± ${{ stage_std }}</p>
                </div>
                {% endif %}
            </div>

            <!-- TAB 2: BATCH SIMULATOR -->
            <div id="tab-batch" class="tab-pane {% if active_tab != 'batch' %}hidden{% endif %}">
                <form method="POST" class="space-y-4">
                    <input type="hidden" name="action_type" value="batch">
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Batch CSV Records Format (Name, Department, Experience, Education, Age, Gender, City)</label>
                        <textarea name="batch_data" rows="5" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-4 text-sm font-mono text-white focus:outline-none focus:border-blue-500" required placeholder="Alex Mercer, Engineering, 7, Master's, 34, Male, San Francisco
Elena Rostova, Data Science, 4, PhD, 29, Female, New York">{{ batch_input_text if batch_input_text else '' }}</textarea>
                    </div>
                    <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-semibold py-3 px-6 rounded-lg transition-colors">Run Batch Process</button>
                </form>

                {% if batch_results %}
                <div class="mt-6 overflow-x-auto">
                    <table class="w-full text-left text-sm text-slate-300">
                        <thead class="bg-slate-800 text-xs uppercase text-slate-400">
                            <tr>
                                <th class="p-3">Name</th>
                                <th class="p-3">Department</th>
                                <th class="p-3">Exp (Yrs)</th>
                                <th class="p-3 text-right">Predicted Value</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-800">
                            {% for row in batch_results %}
                            <tr class="hover:bg-slate-800/50">
                                <td class="p-3 font-medium text-white">{{ row.Name }}</td>
                                <td class="p-3">{{ row.Department }}</td>
                                <td class="p-3">{{ row.Experience_Years }}</td>
                                <td class="p-3 text-right font-bold text-emerald-400">${{ row.Prediction }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
                {% endif %}
            </div>

            <!-- TAB 3: MODEL INSIGHTS -->
            <div id="tab-analytics" class="tab-pane {% if active_tab != 'analytics' %}hidden{% endif %}">
                <div class="space-y-6">
                    <h3 class="text-lg font-semibold text-white">Gradient Boosting Regressor Specifications</h3>
                    <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
                        <div class="bg-slate-800/60 p-4 rounded-xl border border-slate-700">
                            <span class="text-xs text-slate-400 uppercase">Estimators (Trees)</span>
                            <p class="text-2xl font-bold text-blue-400 mt-1">{{ model_info.n_estimators }}</p>
                        </div>
                        <div class="bg-slate-800/60 p-4 rounded-xl border border-slate-700">
                            <span class="text-xs text-slate-400 uppercase">Learning Rate</span>
                            <p class="text-2xl font-bold text-indigo-400 mt-1">{{ model_info.learning_rate }}</p>
                        </div>
                        <div class="bg-slate-800/60 p-4 rounded-xl border border-slate-700">
                            <span class="text-xs text-slate-400 uppercase">Loss Criterion</span>
                            <p class="text-lg font-bold text-purple-400 mt-1">{{ model_info.loss }}</p>
                        </div>
                    </div>
                    <div class="bg-slate-800/40 p-5 rounded-xl border border-slate-800">
                        <h4 class="text-sm font-semibold text-slate-300 mb-2">Required Schema Signature</h4>
                        <p class="text-xs text-slate-400 font-mono">['Name', 'Department', 'Experience_Years', 'Education_Level', 'Age', 'Gender', 'City']</p>
                    </div>
                </div>
            </div>

            {% if error %}
            <div class="mt-6 p-4 bg-red-950/50 border border-red-500/50 text-red-300 rounded-lg text-sm text-center">
                {{ error }}
            </div>
            {% endif %}
        </div>
    </div>

    <script>
        function switchTab(tabId) {
            document.querySelectorAll('.tab-pane').forEach(el => el.classList.add('hidden'));
            document.getElementById('tab-' + tabId).classList.remove('hidden');
            
            ['single', 'batch', 'analytics'].forEach(t => {
                const btn = document.getElementById('btn-' + t);
                if (t === tabId) {
                    btn.classList.add('border-blue-500', 'text-blue-400');
                    btn.classList.remove('border-transparent', 'text-slate-400');
                } else {
                    btn.classList.remove('border-blue-500', 'text-blue-400');
                    btn.classList.add('border-transparent', 'text-slate-400');
                }
            });
        }
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    stage_std = None
    error = None
    employee_name = ""
    form_data = None
    batch_results = None
    batch_input_text = ""
    active_tab = "single"
    model_info = {}

    if model:
        model_info = {
            'n_estimators': getattr(model, 'n_estimators', 'Unknown'),
            'learning_rate': getattr(model, 'learning_rate', 'Unknown'),
            'loss': getattr(model, 'loss', 'Unknown')
        }

    if request.method == 'POST':
        action_type = request.form.get('action_type', 'single')
        active_tab = action_type

        try:
            if model is None:
                raise ValueError("Model object is not loaded. Ensure GradientBoost.pkl is present.")

            if action_type == 'single':
                form_data = request.form
                employee_name = form_data.get('Name', 'Employee')

                input_df = pd.DataFrame([{
                    'Name': employee_name,
                    'Department': form_data.get('Department'),
                    'Experience_Years': float(form_data.get('Experience_Years', 0)),
                    'Education_Level': form_data.get('Education_Level'),
                    'Age': float(form_data.get('Age', 0)),
                    'Gender': form_data.get('Gender'),
                    'City': form_data.get('City')
                }], columns=FEATURE_NAMES)

                raw_predictions = list(model.staged_predict(input_df))
                final_pred = raw_predictions[-1][0]
                std_dev = np.std([p[0] for p in raw_predictions])

                prediction = f"{float(final_pred):,.2f}"
                stage_std = f"{float(std_dev):,.2f}"

            elif action_type == 'batch':
                batch_input_text = request.form.get('batch_data', '')
                rows = []
                for line in batch_input_text.strip().split('\n'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 7:
                        rows.append({
                            'Name': parts[0],
                            'Department': parts[1],
                            'Experience_Years': float(parts[2]),
                            'Education_Level': parts[3],
                            'Age': float(parts[4]),
                            'Gender': parts[5],
                            'City': parts[6]
                        })
                if rows:
                    batch_df = pd.DataFrame(rows, columns=FEATURE_NAMES)
                    preds = model.predict(batch_df)
                    batch_results = []
                    for i, row in enumerate(rows):
                        row['Prediction'] = f"{float(preds[i]):,.2f}"
                        batch_results.append(row)

        except Exception as e:
            error = f"Execution Error: {str(e)}"

    return render_template_string(
        HTML_TEMPLATE,
        prediction=prediction,
        stage_std=stage_std,
        error=error,
        employee_name=employee_name,
        form_data=form_data,
        batch_results=batch_results,
        batch_input_text=batch_input_text,
        active_tab=active_tab,
        model_info=model_info
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)