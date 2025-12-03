angular.module('agentApp', [])
.directive('fileModel', ['$parse', function ($parse) {
    return {
        restrict: 'A',
        link: function(scope, element, attrs) {
            var model = $parse(attrs.fileModel);
            var modelSetter = model.assign;
            
            element.bind('change', function(){
                scope.$apply(function(){
                    modelSetter(scope, element[0].files[0]);
                });
            });
        }
    };
}])
.controller('MainController', function($scope, $http) {
    $scope.agents = [];
    $scope.results = [];
    $scope.columns = [];
    $scope.processing = false;
    $scope.statusMessage = '';
    $scope.statusClass = '';
    $scope.currentFilename = '';
    $scope.showModal = false;
    $scope.showImageModal = false;
    $scope.currentImageUrl = '';
    $scope.currentRowIndex = -1;
    $scope.feedback = {
        text: '',
        score: null,
        page: null,
        extracted_value: null,
        summary: null
    };
    $scope.progress = {}; // Track progress per agent
    $scope.selectedFile = null;
    
    // Load agents on startup
    $http.get('/api/agents')
        .then(function(response) {
            $scope.agents = response.data.agents;
            // Initialize progress
            $scope.agents.forEach(function(agent) {
                $scope.progress[agent] = 0;
            });
        })
        .catch(function(error) {
            console.error('Error loading agents:', error);
            $scope.statusMessage = 'Error loading agents';
            $scope.statusClass = 'error';
        });
    
    $scope.progressInterval = null;
    
    $scope.uploadFile = function() {
        if (!$scope.selectedFile) {
            $scope.statusMessage = 'Please select a file';
            $scope.statusClass = 'error';
            return;
        }
        
        var formData = new FormData();
        formData.append('file', $scope.selectedFile);
        
        $scope.processing = true;
        $scope.statusMessage = 'Uploading and processing...';
        $scope.statusClass = 'processing';
        $scope.results = [];
        
        // Reset progress
        $scope.agents.forEach(function(agent) {
            $scope.progress[agent] = 0;
        });
        
        $http.post('/api/upload', formData, {
            headers: {
                'Content-Type': undefined
            }
        })
        .then(function(response) {
            if (response.data.success) {
                // Start polling for progress
                $scope.startProgressPolling();
            } else {
                $scope.processing = false;
                $scope.statusMessage = 'Processing failed: ' + (response.data.error || 'Unknown error');
                $scope.statusClass = 'error';
            }
        })
        .catch(function(error) {
            $scope.processing = false;
            $scope.statusMessage = 'Error: ' + (error.data?.error || error.message || 'Unknown error');
            $scope.statusClass = 'error';
        });
    };
    
    $scope.startProgressPolling = function() {
        if ($scope.progressInterval) {
            clearInterval($scope.progressInterval);
        }
        
        $scope.progressInterval = setInterval(function() {
            $http.get('/api/progress')
                .then(function(response) {
                    var status = response.data;
                    
                    // Update progress for each agent
                    if (status.progress) {
                        for (var agent in status.progress) {
                            $scope.progress[agent] = status.progress[agent];
                        }
                    }
                    
                    // Check if completed
                    if (status.status === 'completed') {
                        $scope.processing = false;
                        $scope.statusMessage = 'Processing completed!';
                        $scope.statusClass = 'success';
                        $scope.currentFilename = status.filename;
                        clearInterval($scope.progressInterval);
                        $scope.progressInterval = null;
                        // Load results
                        if (status.filename) {
                            $scope.loadResults(status.filename);
                        }
                    } else if (status.status === 'error') {
                        $scope.processing = false;
                        $scope.statusMessage = 'Processing error: ' + (status.error || 'Unknown error');
                        $scope.statusClass = 'error';
                        clearInterval($scope.progressInterval);
                        $scope.progressInterval = null;
                    }
                })
                .catch(function(error) {
                    console.error('Error polling progress:', error);
                });
        }, 1000); // Poll every second
    };
    
    $scope.loadResults = function(filename) {
        $http.get('/api/results/' + filename)
            .then(function(response) {
                $scope.results = response.data.data;
                $scope.columns = response.data.columns;
                
                // Update progress to 100% for all agents
                $scope.agents.forEach(function(agent) {
                    $scope.progress[agent] = 100;
                });
            })
            .catch(function(error) {
                console.error('Error loading results:', error);
                $scope.statusMessage = 'Error loading results';
                $scope.statusClass = 'error';
            });
    };
    
    $scope.getProgress = function(agent) {
        return $scope.progress[agent] || 0;
    };
    
    $scope.openFeedback = function(rowIndex) {
        $scope.currentRowIndex = rowIndex;
        $scope.feedback = {
            text: '',
            score: null,
            page: null,
            extracted_value: null,
            summary: null
        };
        $scope.showModal = true;
    };
    
    $scope.closeModal = function() {
        $scope.showModal = false;
        $scope.currentRowIndex = -1;
    };
    
    $scope.setThumb = function(field, value) {
        $scope.feedback[field] = value;
    };
    
    $scope.submitFeedback = function() {
        if ($scope.currentRowIndex === -1) return;
        
        var row = $scope.results[$scope.currentRowIndex];
        var feedbackData = {
            row_id: $scope.currentRowIndex,
            text: $scope.feedback.text,
            score: $scope.feedback.score,
            page: $scope.feedback.page,
            extracted_value: $scope.feedback.extracted_value,
            summary: $scope.feedback.summary
        };
        
        $http.post('/api/feedback', feedbackData)
            .then(function(response) {
                if (response.data.success) {
                    alert('Feedback submitted successfully!');
                    $scope.closeModal();
                } else {
                    alert('Error submitting feedback');
                }
            })
            .catch(function(error) {
                console.error('Error submitting feedback:', error);
                alert('Error submitting feedback');
            });
    };
    
    $scope.downloadCSV = function() {
        if (!$scope.currentFilename) return;
        
        window.location.href = '/api/download/' + $scope.currentFilename;
    };
    
    $scope.generateReport = function() {
        // Placeholder - no function yet
        alert('Generate Report function not yet implemented');
    };
    
    $scope.showPageImage = function(source) {
        // Extract page number from source (e.g., "page 1" -> 1)
        var match = source.match(/page\s+(\d+)/i);
        if (match) {
            var pageNum = parseInt(match[1]);
            $scope.currentImageUrl = '/api/image/' + pageNum;
            $scope.showImageModal = true;
        }
    };
    
    $scope.closeImageModal = function() {
        $scope.showImageModal = false;
        $scope.currentImageUrl = '';
    };
});

