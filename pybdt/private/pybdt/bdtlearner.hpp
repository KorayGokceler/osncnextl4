// bdtlearner.hpp


#ifndef PYBDT_BDTLEARNER_HPP
#define PYBDT_BDTLEARNER_HPP

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>

#include "boost_python.hpp"

#include "bdtmodel.hpp"
#include "dataset.hpp"
#include "dtlearner.hpp"
#include "learner.hpp"
#include "pruner.hpp"


class BDTLearner : public Learner {
    friend class BDTLearner_pickle_suite;
public:

    // structors

    explicit BDTLearner (const std::vector<std::string>& feature_names,
                         const std::string& weight_name="");

    BDTLearner (const std::vector<std::string>& feature_names,
                const std::string& sig_weight_name,
                const std::string& bg_weight_name);

    // structors for Python

    explicit BDTLearner (const boost::python::list& feature_names,
                         const std::string& weight_name="");

    BDTLearner (const boost::python::list& feature_names,
                const std::string& sig_weight_name,
                const std::string& bg_weight_name);

    virtual ~BDTLearner ();

    // inspectors

    double beta () const;
    double frac_random_events () const;
    int num_trees () const;
    bool use_purity () const;
    bool quiet () const;

    boost::python::list after_pruners () const;
    boost::python::list before_pruners () const;


    // mutators

    void beta (double beta);
    void frac_random_events (double n);
    void num_trees (int n);
    void use_purity (bool purity);
    void quiet (bool val);

    void add_after_pruner (boost::shared_ptr<Pruner> pruner);
    void add_before_pruner (boost::shared_ptr<Pruner> pruner);
    void clear_after_pruners ();
    void clear_before_pruners ();
    void set_defaults ();

    boost::shared_ptr<DTLearner> dtlearner ();

    virtual boost::shared_ptr<Model> train_given_everything (
        const std::vector<Event>& sig, const std::vector<Event>& bg,
        const std::vector<double>& init_sig_weights,
        const std::vector<double>& init_bg_weights) const;

protected:

    boost::shared_ptr<DTLearner> m_dtlearner;

    double m_beta;
    double m_frac_random_events;
    int m_num_trees;
    bool m_use_purity;
    bool m_quiet;

    std::vector<boost::shared_ptr<Pruner> > m_before_pruners;
    std::vector<boost::shared_ptr<Pruner> > m_after_pruners;


};



void export_bdtlearner ();

#endif  /* PYBDT_BDTLEARNER_HPP */
