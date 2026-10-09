// Jenkins pipeline for DeMo_Fixed_Band_Experiment.
//
// Stages: clean old reports -> set up the virtualenv -> lint -> tests (JUnit via pytest) ->
// smoke run (every variant for a few steps on synthetic data, then the analysis) -> an optional
// nightly sweep on TinyShakespeare. Results and the smoke run's results.md are archived.
//
// Needs on the agent: Python 3.11+, network access to download PyTorch (CPU wheels) on first use.
// Plugins: Pipeline, Git, JUnit.

pipeline {
    agent any

    parameters {
        booleanParam(name: 'NIGHTLY', defaultValue: false,
                     description: 'Also run the core grid on TinyShakespeare (hours on a CPU agent)')
        string(name: 'REFERENCE_DEMO', defaultValue: '',
               description: 'Path to a checkout of bloc97/DeMo demo.py, to run the reference comparison test')
    }

    options {
        buildDiscarder(logRotator(numToKeepStr: '30'))
        timeout(time: 60, unit: 'MINUTES')
    }

    environment {
        // Torch threads: keep the agent responsive while tests and the smoke run train small models.
        OMP_NUM_THREADS = '4'
    }

    stages {
        // The workspace is reused between builds (it keeps the virtualenv), so delete the previous
        // build's reports first; otherwise a build that fails early publishes the last build's results.
        stage('Clean reports') {
            steps {
                sh 'rm -rf pytest-junit.xml results/synthetic results/ckpt'
            }
        }

        stage('Virtualenv') {
            steps {
                sh '''
                    python3 -m venv .venv
                    .venv/bin/pip install -q --upgrade pip
                    .venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
                    .venv/bin/pip install -q -r requirements.txt
                '''
            }
        }

        stage('Lint') {
            steps {
                sh '.venv/bin/ruff check experiment tests --select E,F,W --ignore E501'
            }
        }

        stage('Tests') {
            steps {
                sh '''
                    if [ -n "$REFERENCE_DEMO" ]; then export DEMO_REF="$REFERENCE_DEMO"; fi
                    .venv/bin/pytest tests --junitxml=pytest-junit.xml
                '''
            }
        }

        stage('Smoke run') {
            steps {
                sh '.venv/bin/python experiment/sweep.py --grid ci --dataset synthetic'
                sh '.venv/bin/python experiment/analyse.py --dataset synthetic > /dev/null'
            }
            post {
                always { archiveArtifacts artifacts: 'results/synthetic/**', allowEmptyArchive: true }
            }
        }

        stage('Nightly sweep') {
            when { expression { params.NIGHTLY } }
            options { timeout(time: 24, unit: 'HOURS') }
            steps {
                sh '.venv/bin/python experiment/sweep.py --grid core --dataset shakespeare'
                sh '.venv/bin/python experiment/analyse.py --dataset shakespeare > /dev/null'
                archiveArtifacts artifacts: 'results/shakespeare/**'
            }
        }
    }

    post {
        always {
            junit testResults: 'pytest-junit.xml', allowEmptyResults: true
        }
    }
}
